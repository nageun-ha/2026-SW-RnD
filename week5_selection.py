"""
5주차 하이퍼파라미터 튜닝의 탐색 범위와 승자 선택 규칙.

README "5주차 선정 기준 (실험 전 작성)"을 코드로 옮긴 것이다.
run_experiments.py(무엇을 학습할지)와 compare_results.py(결과 표·그림)가 함께 쓴다.
  → 규칙을 한 곳에만 두어, 두 스크립트가 서로 다른 승자를 고르는 일을 막는다.

torch와 config.py를 import하지 않는다.
  compare_results.py는 장치 검사 없이 가볍게 돌아야 하므로 기본값을 이 파일에도 적어 두고,
  run_experiments.py가 시작할 때 config.py와 같은지 확인한다 (check_defaults_match).

용어
  설정(cfg) : (lr, scheduler, weight_decay) 세 값의 묶음. 나머지 학습 조건은 FIXED_SETTINGS로 고정이다.
  A 승자    : lr × scheduler 6조합 중 고른 설정 (wd는 기본값)
  B 승자    : A 승자 위에서 wd 4개 중 고른 설정 = 모델별 "튜닝한 설정"
  최종 설정 : C에서 시드 3개로 4주차 기본 설정보다 "개선"이면 B 승자, 아니면 기본 설정
"""

import json
import math
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
SELECTION_FILE_NAME = "week5_selection.json"  # 직접 지정한 승자를 기록해 compare_results.py가 같은 설정을 쓰게 함

# ---------------------------------------------------------------------
# 기본 설정 (config.py의 LR, SCHEDULER, WEIGHT_DECAY와 같아야 한다)
# ---------------------------------------------------------------------
DEFAULT_LR = 0.01
DEFAULT_SCHEDULER = "none"
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_CFG = (DEFAULT_LR, DEFAULT_SCHEDULER, DEFAULT_WEIGHT_DECAY)

# 모든 5주차 실험에서 고정하는 값 (README "고정하는 것"). 이 값이 다른 실험은 5주차 결과로 읽지 않는다.
FIXED_SETTINGS = {"optimizer": "sgd", "batch_size": 128, "momentum": 0.9, "augment": True}

# ---------------------------------------------------------------------
# 탐색 범위 (README "탐색 순서" 표)
# ---------------------------------------------------------------------
MODELS = ["bn_deep_wide", "bn_deep"]
EPOCHS = 40
SEARCH_SEED = 42                 # A, B는 이 시드 하나로 비교한다
C_SEEDS = [1, 2]                 # C에서 추가하는 시드
ALL_SEEDS = [SEARCH_SEED] + C_SEEDS
D_EPOCHS = 60                    # D(선택): 수렴 확인용
STAGE_A_LRS = [0.1, 0.05, 0.01]
STAGE_A_SCHEDULERS = ["none", "cosine"]
STAGE_B_WEIGHT_DECAYS = [1e-4, 5e-4, 1e-3, 5e-3]  # 5e-4는 A 결과를 다시 쓴다

# ---------------------------------------------------------------------
# 판정 기준 (README에 실험 전에 고정한 값. 결과를 본 뒤 바꾸지 않는다)
# ---------------------------------------------------------------------
DIFF_THRESHOLD_PP = 0.45      # val 5,000장 표본 오차 √(0.89×0.11/5000) ≈ 0.44%p
LAST_N_EPOCHS = 5             # 주 지표 last5 = 마지막 5 epoch(36~40) val acc 평균
NOT_CONVERGED_MIN_SEEDS = 2   # 시드 3개 중 이 수 이상에서 best epoch이 마지막 5 epoch 안이면 "아직 수렴 안 됨"
DIVERGED_ACC_PCT = 15.0       # last5가 이보다 낮거나 val loss가 NaN이면 "발산" 표시 (무작위로 찍으면 10%)
FLOAT_TOL = 1e-9              # lr, wd 같은 실수 비교 허용 오차 (json 저장 과정의 표기 차이 흡수)


# =====================================================================
# 작은 도우미
# =====================================================================
def same(a, b):
    """실수 두 개가 같은지 (0.1과 0.1000000001 같은 표기 차이는 같다고 봄)."""
    return math.isclose(float(a), float(b), rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)


def cfg_equal(a, b):
    """설정 두 개 (lr, scheduler, wd)가 같은지."""
    return same(a[0], b[0]) and a[1] == b[1] and same(a[2], b[2])


def cfg_label(cfg):
    """설정을 사람이 읽기 좋은 문자열로. 예: 'lr 0.1 + cosine + wd 0.0005'"""
    return f"lr {cfg[0]:g} + {cfg[1]} + wd {cfg[2]:g}"


def cfg_dict(cfg):
    return {"lr": cfg[0], "scheduler": cfg[1], "weight_decay": cfg[2]}


def check_defaults_match(config_module):
    """이 파일의 기본값·고정값이 config.py와 같은지 확인한다. 다른 항목 목록을 돌려준다 (빈 목록이면 정상)."""
    pairs = [
        ("LR", DEFAULT_LR, config_module.LR),
        ("SCHEDULER", DEFAULT_SCHEDULER, config_module.SCHEDULER),
        ("WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY, config_module.WEIGHT_DECAY),
        ("OPTIMIZER", FIXED_SETTINGS["optimizer"], config_module.OPTIMIZER),
        ("BATCH_SIZE", FIXED_SETTINGS["batch_size"], config_module.BATCH_SIZE),
        ("MOMENTUM", FIXED_SETTINGS["momentum"], config_module.MOMENTUM),
        ("USE_AUGMENTATION", FIXED_SETTINGS["augment"], config_module.USE_AUGMENTATION),
    ]
    return [f"{name}: week5_selection {mine!r} / config {theirs!r}" for name, mine, theirs in pairs if mine != theirs]


# =====================================================================
# 결과 읽기
# =====================================================================
def load_index(results_dir=RESULTS_DIR):
    """
    results/에서 5주차 후보 모델의 실험을 모두 읽는다. (4주차 기본 설정 결과도 여기에 포함된다)

    폴더 이름이 아니라 args.json의 값으로 설정을 찾는다.
      → 이름 규칙이 바뀌거나 --run-name으로 이름을 바꿔도 같은 설정을 정확히 찾을 수 있다.
    args.json에 scheduler가 없으면(4주차 이전 실험) none으로 본다.
    FIXED_SETTINGS와 다른 실험(예: batch size를 바꾼 실험)은 읽지 않는다.
    """
    records = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        args_path, history_path = run_dir / "args.json", run_dir / "history.csv"
        if not (run_dir.is_dir() and args_path.exists() and history_path.exists()):
            continue
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if args.get("model") not in MODELS:
            continue
        if any(args.get(key) != value for key, value in FIXED_SETTINGS.items()):
            continue
        try:
            history = pd.read_csv(history_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        epochs = args.get("epochs")
        records.append({
            "name": run_dir.name, "model": args["model"], "seed": args["seed"], "epochs": epochs,
            "cfg": (float(args["lr"]), args.get("scheduler", DEFAULT_SCHEDULER), float(args["weight_decay"])),
            "complete": list(history["epoch"]) == list(range(1, epochs + 1)),
            "device": args.get("device"), "torch_version": args.get("torch_version"),
            "history": history,
        })
    return records


def find_run(index, model, cfg, seed, epochs=EPOCHS):
    """
    (모델, 설정, 시드, epoch)에 해당하는 실험 하나를 찾는다. 없으면 None.
    같은 조건이 2개 이상이면(폴더를 복사한 경우 등) 어느 것을 쓸지 정할 수 없으므로 에러로 멈춘다.
    """
    matches = [r for r in index if r["model"] == model and r["seed"] == seed
               and r["epochs"] == epochs and cfg_equal(r["cfg"], cfg)]
    if len(matches) > 1:
        raise ValueError(f"같은 설정의 실험이 {len(matches)}개 있습니다: {[r['name'] for r in matches]}")
    return matches[0] if matches else None


def run_metrics(record):
    """실험 하나의 지표 (4주차 compare_results.py와 같은 정의)."""
    h = record["history"]
    last5 = h["val_acc"].tail(LAST_N_EPOCHS).mean() * 100
    best_idx = h["val_acc"].idxmax() if h["val_acc"].notna().any() else h.index[0]
    return {
        "last5_val_acc_pct": last5,
        "best_val_acc_pct": h["val_acc"].max() * 100,
        "best_epoch": int(h.loc[best_idx, "epoch"]),  # 동점이면 처음 나온 epoch
        "overfit_gap_pp": (h["train_acc"].iloc[-1] - h["val_acc"].iloc[-1]) * 100,
        "epoch_time_sec": h["epoch_time_sec"].mean(),
        "diverged": bool(h["val_loss"].isna().any() or pd.isna(last5) or last5 < DIVERGED_ACC_PCT),
    }


# =====================================================================
# 승자 선택
# =====================================================================
def closeness_key(cfg):
    """
    "기본값에 가까운 정도" 정렬 키 (작을수록 가깝다). README의 판정 순서와 같다.
      1) 기본값과 다른 항목 수  2) lr의 배수 차이  3) wd의 배수 차이  4) scheduler가 none이 아니면 1
    배수 차이를 log로 재는 이유: 0.05는 0.01의 5배, 0.1은 10배이므로 0.05가 더 가깝다고 보기 위함이다.
    """
    lr, scheduler, weight_decay = cfg
    n_changed = (not same(lr, DEFAULT_LR)) + (scheduler != DEFAULT_SCHEDULER) + (not same(weight_decay, DEFAULT_WEIGHT_DECAY))
    return (n_changed,
            abs(math.log(lr / DEFAULT_LR)),
            abs(math.log(weight_decay / DEFAULT_WEIGHT_DECAY)),
            int(scheduler != DEFAULT_SCHEDULER))


def pick_winner(rows):
    """
    완료된 실험 행들에서 승자를 고른다. rows의 각 행에 판정 결과 열을 채워 넣는다.
      1) 주 지표(last5) 1위와의 차이가 DIFF_THRESHOLD_PP 이하면 "1위와 차이 없음" 그룹 (1위 포함)
      2) 그룹 안에서 기본값에 가장 가까운 설정 (closeness_key)
    발산한 실험(last5가 NaN)은 가장 낮은 점수로 본다.
    """
    def score(row):
        value = row["last5_val_acc_pct"]
        return -math.inf if pd.isna(value) else value

    top = max(score(row) for row in rows)
    if top == -math.inf:
        raise ValueError("모든 실험이 발산해 승자를 고를 수 없습니다.")
    for row in rows:
        row["diff_vs_top_pp"] = score(row) - top if score(row) > -math.inf else math.nan
        row["same_as_top"] = top - score(row) <= DIFF_THRESHOLD_PP + FLOAT_TOL
    group = [row for row in rows if row["same_as_top"]]
    winner = min(group, key=lambda row: closeness_key((row["lr"], row["scheduler"], row["weight_decay"])))
    for row in rows:
        row["winner"] = row is winner
    return (winner["lr"], winner["scheduler"], winner["weight_decay"])


def stage_configs(stage, base_cfg=None):
    """단계 A, B에서 비교할 설정 목록."""
    if stage == "A":
        return [(lr, scheduler, DEFAULT_WEIGHT_DECAY) for lr in STAGE_A_LRS for scheduler in STAGE_A_SCHEDULERS]
    if stage == "B":
        return [(base_cfg[0], base_cfg[1], wd) for wd in STAGE_B_WEIGHT_DECAYS]
    raise ValueError(f"설정 목록이 있는 단계는 A, B입니다: {stage}")


def reuse_source(stage, cfg, seed, epochs=EPOCHS):
    """
    이 (설정, 시드)가 이미 다른 단계에서 학습된 결과를 다시 쓰는 것이면 그 출처를, 아니면 None을 돌려준다.
    다시 쓰는 실험은 새로 학습하지 않는다 (run_experiments.py가 결과가 없어도 학습하지 않고 알려 줌).
    """
    if epochs == EPOCHS and cfg_equal(cfg, DEFAULT_CFG):
        return "4주차 결과 재사용"
    if stage == "B" and same(cfg[2], DEFAULT_WEIGHT_DECAY):
        return "A 결과 재사용"
    if stage == "C" and seed == SEARCH_SEED:
        return "A·B 결과 재사용"
    return None


def stage_rows(index, model, stage, base_cfg=None):
    """
    단계 A 또는 B의 표(행 목록)를 만들고 승자를 고른다.
    반환: (행 목록, 승자 설정 또는 None, 고를 수 없는 이유 또는 None)
    행의 '_history'는 그림용 내부 값이라 CSV에는 저장하지 않는다.
    """
    rows = []
    for cfg in stage_configs(stage, base_cfg):
        record = find_run(index, model, cfg, SEARCH_SEED)
        row = {"model": model, "stage": stage, **cfg_dict(cfg), "seed": SEARCH_SEED,
               "run": record["name"] if record else "",
               "source": reuse_source(stage, cfg, SEARCH_SEED) or f"5주차 {stage}",
               "device": record["device"] if record else "", "_history": None}
        if record is None:
            row["status"] = "없음"
        elif not record["complete"]:
            row["status"] = "미완료"
        else:
            row["status"] = "완료"
            row.update(run_metrics(record))
            row["_history"] = record["history"]
        rows.append(row)
    unfinished = [row for row in rows if row["status"] != "완료"]
    if unfinished:
        names = ", ".join(cfg_label((r["lr"], r["scheduler"], r["weight_decay"])) for r in unfinished)
        return rows, None, f"{stage} 실험 {len(unfinished)}/{len(rows)}개가 끝나지 않음 ({names})"
    return rows, pick_winner(rows), None


# =====================================================================
# 직접 지정한 승자 기록 (results/week5_selection.json)
# =====================================================================
def load_selection(results_dir=RESULTS_DIR):
    """기록된 승자를 읽는다. 형식: {모델: {"A": {lr, scheduler, weight_decay, source}, "B": {...}}}"""
    path = Path(results_dir) / SELECTION_FILE_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_selection(selection, results_dir=RESULTS_DIR):
    path = Path(results_dir) / SELECTION_FILE_NAME
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _saved_cfg(entry):
    return (float(entry["lr"]), entry["scheduler"], float(entry["weight_decay"]))


def resolve_winners(index, model, saved=None, override_a=None, override_wd=None):
    """
    모델의 A 승자, B 승자를 정한다. 우선순위: 이번에 직접 지정한 값 > 기록 파일의 직접 지정 > 결과로 자동 선택.
      override_a : (lr, scheduler)  → A 승자를 직접 지정 (wd는 기본값)
      override_wd: wd                → B 승자의 wd를 직접 지정 (lr, scheduler는 A 승자)
    자동으로 고를 수 없으면(앞 단계 실험이 덜 끝남) 승자는 None이고 이유를 'problem'에 적는다.
    """
    saved = (saved or {}).get(model, {})
    out = {"A": None, "A_source": None, "B": None, "B_source": None, "problem": None,
           "A_rows": None, "B_rows": None}

    rows_a, auto_a, problem_a = stage_rows(index, model, "A")
    out["A_rows"] = rows_a
    if override_a is not None:
        out["A"], out["A_source"] = (float(override_a[0]), override_a[1], DEFAULT_WEIGHT_DECAY), "직접 지정"
    elif saved.get("A", {}).get("source") == "직접 지정":
        out["A"], out["A_source"] = _saved_cfg(saved["A"]), "직접 지정 (기록 파일)"
    elif auto_a is not None:
        out["A"], out["A_source"] = auto_a, "자동"
    else:
        out["problem"] = problem_a
        return out

    rows_b, auto_b, problem_b = stage_rows(index, model, "B", out["A"])
    out["B_rows"] = rows_b
    saved_b = saved.get("B", {})
    if override_wd is not None:
        out["B"], out["B_source"] = (out["A"][0], out["A"][1], float(override_wd)), "직접 지정"
    # 기록된 B 지정값은 지금의 A 승자 위에서 정한 것일 때만 쓴다 (A가 바뀌었으면 무효)
    elif (saved_b.get("source") == "직접 지정" and same(saved_b["lr"], out["A"][0])
          and saved_b["scheduler"] == out["A"][1]):
        out["B"], out["B_source"] = _saved_cfg(saved_b), "직접 지정 (기록 파일)"
    elif auto_b is not None:
        out["B"], out["B_source"] = auto_b, "자동"
    else:
        out["problem"] = problem_b
    return out


# =====================================================================
# C: 시드 3개로 최종 설정 판정, D 필요 여부
# =====================================================================
def seed_summary(index, model, cfg, seeds=ALL_SEEDS, epochs=EPOCHS):
    """한 설정의 시드별 지표와 평균·표본 표준편차(ddof=1)."""
    per_seed = {}
    for seed in seeds:
        record = find_run(index, model, cfg, seed, epochs)
        if record is not None and record["complete"]:
            per_seed[seed] = run_metrics(record)
    table = pd.DataFrame(per_seed).T if per_seed else pd.DataFrame()
    summary = {"n_seeds": len(per_seed), "missing_seeds": [s for s in seeds if s not in per_seed], "per_seed": per_seed}
    for col in ("last5_val_acc_pct", "best_val_acc_pct", "overfit_gap_pp"):
        values = table[col].astype(float) if per_seed else pd.Series(dtype=float)
        summary[f"{col}_mean"] = values.mean() if len(values) else math.nan
        summary[f"{col}_std"] = values.std() if len(values) > 1 else math.nan
    summary["best_epochs"] = [per_seed[s]["best_epoch"] for s in seeds if s in per_seed]
    return summary


def is_not_converged(best_epochs, epochs=EPOCHS):
    """4주차와 같은 판정: 2개 이상 시드에서 best epoch이 마지막 5 epoch(36~40) 안이면 True."""
    return sum(e >= epochs - LAST_N_EPOCHS + 1 for e in best_epochs) >= NOT_CONVERGED_MIN_SEEDS


def judge_final(index, model, tuned_cfg):
    """
    C 판정: B 승자(튜닝한 설정)의 시드 3개를 4주차 기본 설정의 시드 3개와 비교해 최종 설정을 정한다.
      개선 = (평균 차이 > DIFF_THRESHOLD_PP) 그리고 (평균 차이 > 두 설정의 시드 표준편차 중 큰 값)
      개선이면 튜닝한 설정, 아니면 기본 설정이 최종 설정이다.
    D 필요 여부는 '최종 설정'의 시드 3개 best epoch으로 판정한다.
    """
    tuned = seed_summary(index, model, tuned_cfg)
    default = seed_summary(index, model, DEFAULT_CFG)
    result = {"model": model, "tuned": tuned, "default": default, "tuned_cfg": tuned_cfg,
              "diff_pp": math.nan, "larger_seed_std_pp": math.nan, "improved": None, "chosen": None}
    n_seeds = len(ALL_SEEDS)
    if cfg_equal(tuned_cfg, DEFAULT_CFG):
        result["judgment"] = "B 승자가 기본 설정 → 기본 설정 유지"
        result["chosen"] = DEFAULT_CFG
    elif tuned["n_seeds"] < n_seeds or default["n_seeds"] < n_seeds:
        result["judgment"] = (f"C 미완료 (튜닝 설정 시드 {tuned['n_seeds']}/{n_seeds}, "
                              f"기본 설정 시드 {default['n_seeds']}/{n_seeds})")
        return result
    else:
        diff = tuned["last5_val_acc_pct_mean"] - default["last5_val_acc_pct_mean"]
        larger_std = max(tuned["last5_val_acc_pct_std"], default["last5_val_acc_pct_std"])
        improved = diff > DIFF_THRESHOLD_PP and diff > larger_std
        result.update(diff_pp=diff, larger_seed_std_pp=larger_std, improved=improved,
                      chosen=tuned_cfg if improved else DEFAULT_CFG,
                      judgment="개선 → 튜닝한 설정 채택" if improved else "개선 아님 → 기본 설정 유지")
    chosen = tuned if cfg_equal(result["chosen"], tuned_cfg) else default
    result["not_converged"] = chosen["n_seeds"] == n_seeds and is_not_converged(chosen["best_epochs"])
    result["d_needed"] = result["not_converged"]
    return result
