"""
저장된 체크포인트로 Grad-CAM 그림을 그린다. 학습은 하지 않는다.

실행 예:
    python visualize_gradcam.py --mode check
    python visualize_gradcam.py --mode single --runs baseline_sgd_lr0.01_bs128_ep20_aug_seed42
    python visualize_gradcam.py --mode compare --runs <run1> <run2> --per-class 1
    python visualize_gradcam.py --mode misclassified --runs <run> --pairs cat,dog automobile,truck --max-per-pair 8
    python visualize_gradcam.py --mode single --runs <run> --indices figures/gradcam/<이전 결과>_indices.csv

결과: figures/gradcam/<모드>_<val|test>_<모델들>.png, 같은 이름의 _indices.csv
    (misclassified는 _confusion.png, _confusion.csv도 저장)

※ 기본 데이터는 validation이다. 테스트셋은 --test를 줄 때만 쓰며 6주차 최종 평가 전용이다.
※ 그림 안의 글자는 영어로 쓴다. matplotlib 기본 폰트에 한글이 없어 네모로 깨지기 때문이다.
"""

import argparse
import hashlib
import math
import sys

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

import config
from data import (get_class_names, get_dataloaders, get_norm_stats, get_split,
                  get_test_loader, set_seed)
from eda import denormalize
from engine import confusion_matrix, evaluate, make_criterion
from evaluate import load_model, save_confusion_matrix_figure
from gradcam import GradCAM, disable_inplace_relu, find_target_layer
from models import get_model, list_models

OUTPUT_DIR = config.FIG_DIR / "gradcam"
DEFAULT_PAIRS = ["cat,dog", "automobile,truck"]
DEFAULT_PER_CLASS = 1
DEFAULT_MAX_PER_PAIR = 8
MAX_RUNS_NAME_LEN = 120     # Windows 경로 길이 제한(260자)을 넘지 않도록 파일 이름의 실험 부분을 제한
RUNS_HASH_LEN = 8

# 그림 설정. inferno는 밝기가 한 방향으로 변해 없는 경계가 생기지 않고 색약에도 비교적 안전하다.
CMAP = "inferno"
OVERLAY_ALPHA = 0.5         # 오버레이에서 히트맵 비중
PANEL_WIDTH = 2.2           # 칸 하나의 가로 크기(인치)
PANEL_HEIGHT = 2.4          # 칸 하나의 세로 크기(인치)
TITLE_FONT_SIZE = 8
COLORBAR_ROWS = 3           # 색상 막대의 최대 높이(행 수 기준)

# --mode check 설정. 체크포인트가 팀원 PC에 있으면 --runs로 다른 실험을 지정할 수 있다.
CHECK_RUN = "baseline_sgd_lr0.01_bs128_ep20_aug_seed42"
EXPECTED_CAM_SOURCE_SIZE = 8     # 자동 선택 층의 특징맵 크기 (마지막 MaxPool 전)
MAX_ALL_ZERO_RATIO = 0.05        # CAM이 전부 0인 이미지 비율의 허용치
MIN_TARGET_CHANGE_DIFF = 0.01    # target class를 바꿨을 때 CAM 평균 절대 차이의 최소값
MIN_TRAINED_RANDOM_DIFF = 0.05   # 학습 모델과 랜덤 모델의 CAM 평균 절대 차이의 최소값
MAX_VAL_ACC_DIFF = 0.0002        # val 5,000장 중 1장에 해당하는 정확도 차이
MIN_GAP_CORRELATION = 0.999      # gap 모델의 Grad-CAM과 원래 CAM의 최소 상관계수


# =====================================================================
# 인자 처리
# =====================================================================
def parse_args():
    """커맨드라인 인자를 읽고, 모드와 맞지 않는 조합을 실행 전에 막는다."""
    parser = argparse.ArgumentParser(description="Grad-CAM 시각화 (학습 없음)")
    parser.add_argument("--mode", required=True, choices=["single", "compare", "misclassified", "check"])
    parser.add_argument("--runs", nargs="+", default=None, help="results/ 아래 실험 폴더 이름")
    parser.add_argument("--test", action="store_true", help="테스트셋 사용 (6주차 최종 평가 전용!)")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS, help="클래스별 이미지 수")
    parser.add_argument("--indices", nargs="+", default=None,
                        help="이미지 위치 목록(val이면 0~4999) 또는 이전에 저장한 _indices.csv 경로")
    parser.add_argument("--layer", type=int, default=None,
                        help="features 안의 층 인덱스 (생략하면 마지막 Conv 다음 ReLU 자동 선택)")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="헷갈린 클래스 쌍, 예: cat,dog")
    parser.add_argument("--max-per-pair", type=int, default=DEFAULT_MAX_PER_PAIR)
    args = parser.parse_args()

    if args.mode == "check":
        if args.test:  # 검증은 테스트셋을 볼 이유가 없으므로 아예 막는다
            parser.error("--mode check는 validation으로만 실행합니다. --test를 빼 주세요.")
    elif args.runs is None:
        parser.error(f"--mode {args.mode}에는 --runs가 필요합니다.")
    elif args.mode == "compare":
        if len(args.runs) < 2:
            parser.error("--mode compare에는 실험을 2개 이상 지정해 주세요.")
    elif len(args.runs) != 1:
        parser.error(f"--mode {args.mode}에는 실험을 1개만 지정해 주세요.")
    if args.per_class < 1 or args.max_per_pair < 1:
        parser.error("--per-class와 --max-per-pair는 1 이상이어야 합니다.")
    return args


# =====================================================================
# 데이터 / 모델 준비
# =====================================================================
def load_split(use_test):
    """사용할 데이터 로더와 split 이름을 돌려준다. 기본은 validation이다."""
    if use_test:
        print("[경고] 테스트셋을 사용합니다. 최종 평가용이며 모델 선택·튜닝이 끝난 6주차에만 사용하세요.")
        return get_test_loader(confirm_final_eval=True), "test"
    _, val_loader = get_dataloaders(augment=False)
    return val_loader, "val"


def split_labels(dataset):
    """
    이미지를 읽지 않고 라벨과 CIFAR-10 원본 인덱스를 구한다.

    Returns:
        labels: 데이터셋 위치 순서의 정답 라벨, cifar_indices: 각 위치의 원본 인덱스
    """
    if isinstance(dataset, Subset):  # val은 Subset, test는 CIFAR10 그대로다
        base, indices = dataset.dataset, np.asarray(dataset.indices)
    else:
        base, indices = dataset, np.arange(len(dataset))
    return np.asarray(base.targets)[indices], indices


def select_per_class(labels, per_class):
    """클래스별로 per_class장을 고른다. 시드를 고정해 모델과 상관없이 항상 같은 이미지가 나온다."""
    rng = np.random.default_rng(config.SEED)
    positions = []
    for class_index in range(config.NUM_CLASSES):
        candidates = np.flatnonzero(labels == class_index)
        positions.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return positions


def parse_indices(values, split_name, dataset_size):
    """
    --indices를 위치 목록으로 바꾼다. 숫자들 또는 저장된 CSV 경로 하나를 받는다.

    CSV의 split이 현재 실행과 다르면 멈춘다.
    같은 번호라도 val과 test는 다른 이미지이므로, 조용히 다른 그림이 나오는 일을 막기 위함이다.
    """
    if len(values) == 1 and values[0].lower().endswith(".csv"):
        table = pd.read_csv(values[0])
        csv_splits = set(table["split"])
        if csv_splits != {split_name}:
            raise SystemExit(f"CSV의 split({sorted(csv_splits)})이 현재 실행({split_name})과 다릅니다.")
        positions = table["position"].astype(int).tolist()
    else:
        try:
            positions = [int(value) for value in values]
        except ValueError:
            raise SystemExit(f"--indices는 정수 목록 또는 .csv 경로여야 합니다: {values}")
    out_of_range = [p for p in positions if not 0 <= p < dataset_size]
    if out_of_range:
        raise SystemExit(f"--indices 범위(0~{dataset_size - 1})를 벗어났습니다: {out_of_range}")
    return positions


def load_images(dataset, positions):
    """지정한 위치의 정규화된 이미지를 (N, C, H, W) 텐서로 읽는다."""
    return torch.stack([dataset[position][0] for position in positions])


def load_run(run_name, device):
    """
    실험 폴더에서 모델과 체크포인트 정보를 읽는다. 모델 복원은 evaluate.load_model을 재사용한다.
    load_model은 모델만 돌려주므로 val_acc와 설정을 보려면 체크포인트를 한 번 더 읽는다.
    """
    run_dir = config.RESULTS_DIR / run_name
    if not (run_dir / "best.pt").exists():
        raise SystemExit(f"체크포인트가 없습니다: {run_dir / 'best.pt'}\n"
                         ".pt 파일은 git에 올라가지 않으므로 학습한 팀원의 PC에 있을 수 있습니다.")
    model = load_model(run_dir, device)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    info = {key: checkpoint[key] for key in ("model_name", "epoch", "val_acc", "args")}
    return model, info


def compute_cams(cam, images, device, targets=None):
    """
    이미지를 배치로 나눠 Grad-CAM을 계산한다.

    Returns:
        cams(N, H, W), probs(N, num_classes), used_targets(N,): 모두 CPU 텐서
    """
    cams, probs, used = [], [], []
    for start in range(0, len(images), config.EVAL_BATCH_SIZE):
        batch = images[start:start + config.EVAL_BATCH_SIZE].to(device)
        batch_targets = None if targets is None else targets[start:start + config.EVAL_BATCH_SIZE]
        batch_cams, logits, batch_used = cam(batch, batch_targets)
        cams.append(batch_cams.cpu())
        probs.append(logits.softmax(dim=1).cpu())  # confidence 표시용
        used.append(batch_used.cpu())
    return torch.cat(cams), torch.cat(probs), torch.cat(used)


def describe_layer(run_name, cam):
    """실제로 고른 층과 CAM 원본 크기를 출력한다. --layer 번호는 모델마다 다른 층일 수 있어 확인이 필요하다."""
    layer = cam.model.features[cam.layer_index]
    height, width = cam.activation.shape[-2:]
    print(f"  {run_name}: features[{cam.layer_index}] = {type(layer).__name__}, CAM 원본 크기 {height}×{width}")


# =====================================================================
# 저장 도우미
# =====================================================================
def output_stem(mode, split_name, runs, layer):
    """
    출력 파일의 공통 이름을 만든다.
    실험 이름이 많으면 경로가 너무 길어지므로 줄이고, 전체 목록은 CSV의 runs 열에 남긴다.
    """
    joined = "+".join(runs)
    if len(joined) > MAX_RUNS_NAME_LEN:
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:RUNS_HASH_LEN]
        joined = f"{runs[0]}_and_{len(runs) - 1}_more_{digest}"
    suffix = "" if layer is None else f"_layer{layer}"  # 다른 층의 결과가 기본 결과를 덮어쓰지 않게 한다
    return f"{mode}_{split_name}_{joined}{suffix}"


def save_indices(stem, split_name, runs, positions, cifar_indices, labels, class_names, extra=None):
    """선택한 이미지를 CSV로 저장한다. --indices에 그대로 넘기면 같은 이미지를 다시 쓸 수 있다."""
    table = pd.DataFrame({
        "split": split_name,
        "position": positions,
        "cifar_index": cifar_indices[positions],
        "true_class": [class_names[labels[p]] for p in positions],
        "runs": ";".join(runs),
    })
    for name, values in (extra or {}).items():
        table[name] = values
    path = OUTPUT_DIR / f"{stem}_indices.csv"
    table.to_csv(path, index=False)
    print(f"  → 저장: {path.relative_to(config.PROJECT_DIR)}")


def save_figure(fig, stem):
    """그림을 저장하고 메모리에서 닫는다."""
    path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"  → 저장: {path.relative_to(config.PROJECT_DIR)}")


def make_grid(n_rows, n_cols):
    """이미지 격자를 만들고 축 눈금을 지운다."""
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="constrained",
                             figsize=(n_cols * PANEL_WIDTH, n_rows * PANEL_HEIGHT))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def overlay(image, cam_map):
    """원본(0~1) 위에 히트맵을 일정 비율로 겹친다."""
    heat = plt.get_cmap(CMAP)(cam_map)[..., :3]
    return np.clip((1 - OVERLAY_ALPHA) * image + OVERLAY_ALPHA * heat, 0, 1)


def show_heatmap(ax, cam_map):
    """0~1 범위를 고정해 히트맵을 그린다. 칸끼리 같은 색이 같은 값을 뜻하게 하기 위함이다."""
    return ax.imshow(cam_map, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")


def add_colorbar(fig, mappable, axes):
    """
    색이 뜻하는 값을 알 수 있도록 격자 오른쪽 끝에 색상 막대를 붙인다.
    행이 많아도 막대가 COLORBAR_ROWS행 높이를 넘지 않게 해서 이미지 칸의 크기를 지킨다.
    """
    n_rows = axes.shape[0]
    fig.colorbar(mappable, ax=axes, shrink=min(1.0, COLORBAR_ROWS / n_rows), label="Grad-CAM (0-1)")


# =====================================================================
# 모드 (a) single
# =====================================================================
def run_single(args, device):
    """모델 1개에 대해 이미지마다 [원본 | 히트맵 | 오버레이]를 그린다."""
    loader, split_name = load_split(args.test)
    dataset = loader.dataset
    labels, cifar_indices = split_labels(dataset)
    positions = (parse_indices(args.indices, split_name, len(dataset)) if args.indices
                 else select_per_class(labels, args.per_class))
    class_names = get_class_names()
    mean, std = get_norm_stats(get_split()["train_indices"])
    images = load_images(dataset, positions)

    run_name = args.runs[0]
    model, _ = load_run(run_name, device)
    with GradCAM(model, args.layer) as cam:
        cams, probs, preds = compute_cams(cam, images, device)
        describe_layer(run_name, cam)

    fig, axes = make_grid(len(positions), 3)
    mappable = None
    for row, position in enumerate(positions):
        image = denormalize(images[row], mean, std)
        cam_map = cams[row].numpy()
        pred = preds[row].item()
        true_name = class_names[labels[position]]
        mark = "" if pred == labels[position] else " [wrong]"  # 색이 아니라 글자로 오답을 표시
        axes[row, 0].imshow(image, interpolation="nearest")
        axes[row, 0].set_title(f"#{position} true: {true_name}", fontsize=TITLE_FONT_SIZE)
        mappable = show_heatmap(axes[row, 1], cam_map)
        axes[row, 1].set_title(f"pred: {class_names[pred]} ({probs[row, pred]:.2f}){mark}",
                               fontsize=TITLE_FONT_SIZE)
        axes[row, 2].imshow(overlay(image, cam_map), interpolation="nearest")
        axes[row, 2].set_title("overlay", fontsize=TITLE_FONT_SIZE)
    add_colorbar(fig, mappable, axes)

    stem = output_stem("single", split_name, args.runs, args.layer)
    save_figure(fig, stem)
    save_indices(stem, split_name, args.runs, positions, cifar_indices, labels, class_names)


# =====================================================================
# 모드 (b) compare
# =====================================================================
def run_compare(args, device):
    """같은 이미지(행)를 여러 모델(열)로 비교한다. 첫 열은 원본이다."""
    loader, split_name = load_split(args.test)
    dataset = loader.dataset
    labels, cifar_indices = split_labels(dataset)
    positions = (parse_indices(args.indices, split_name, len(dataset)) if args.indices
                 else select_per_class(labels, args.per_class))
    class_names = get_class_names()
    mean, std = get_norm_stats(get_split()["train_indices"])
    images = load_images(dataset, positions)

    fig, axes = make_grid(len(positions), 1 + len(args.runs))
    for row, position in enumerate(positions):
        axes[row, 0].imshow(denormalize(images[row], mean, std), interpolation="nearest")
        axes[row, 0].set_title(f"#{position} true: {class_names[labels[position]]}", fontsize=TITLE_FONT_SIZE)

    for col, run_name in enumerate(args.runs, start=1):
        model, info = load_run(run_name, device)
        with GradCAM(model, args.layer) as cam:
            cams, probs, preds = compute_cams(cam, images, device)
            describe_layer(run_name, cam)
        # 실험 이름 전체는 칸보다 길어서 구별에 필요한 값만 첫 줄에 적는다
        header = f"{info['model_name']} ep{info['args']['epochs']} seed{info['args']['seed']}"
        for row, position in enumerate(positions):
            pred = preds[row].item()
            mark = "" if pred == labels[position] else " [wrong]"
            axes[row, col].imshow(overlay(denormalize(images[row], mean, std), cams[row].numpy()),
                                  interpolation="nearest")
            prefix = f"{header}\n" if row == 0 else ""
            axes[row, col].set_title(f"{prefix}{class_names[pred]} ({probs[row, pred]:.2f}){mark}",
                                     fontsize=TITLE_FONT_SIZE)
        del model

    stem = output_stem("compare", split_name, args.runs, args.layer)
    save_figure(fig, stem)
    save_indices(stem, split_name, args.runs, positions, cifar_indices, labels, class_names)


# =====================================================================
# 모드 (c) misclassified
# =====================================================================
def parse_pairs(pairs, class_names):
    """'cat,dog' 형식의 쌍을 클래스 번호로 바꾼다."""
    parsed = []
    for pair in pairs:
        names = [name.strip() for name in pair.split(",")]
        if len(names) != 2 or names[0] == names[1]:
            raise SystemExit(f"클래스 쌍은 'cat,dog'처럼 서로 다른 2개를 적어 주세요: {pair}")
        unknown = [name for name in names if name not in class_names]
        if unknown:
            raise SystemExit(f"없는 클래스 이름입니다: {unknown}. 사용 가능: {class_names}")
        parsed.append((class_names.index(names[0]), class_names.index(names[1])))
    return parsed


def select_confused(preds, labels, first, second, max_count, rng):
    """
    first와 second를 서로 헷갈린 샘플을 고른다.
    max_count를 양방향에 반씩 나누고, 한쪽이 부족하면 다른 쪽에서 채운다.
    """
    forward = np.flatnonzero((labels == first) & (preds == second))
    backward = np.flatnonzero((labels == second) & (preds == first))
    n_forward = min(len(forward), max(max_count - len(backward), math.ceil(max_count / 2)))
    n_backward = min(len(backward), max_count - n_forward)
    chosen = np.concatenate([rng.choice(forward, size=n_forward, replace=False),
                             rng.choice(backward, size=n_backward, replace=False)])
    return chosen.astype(int).tolist()


def run_misclassified(args, device):
    """헷갈린 샘플을 [원본 | 예측 클래스 CAM | 정답 클래스 CAM]으로 그리고 혼동행렬도 저장한다."""
    loader, split_name = load_split(args.test)
    dataset = loader.dataset
    class_names = get_class_names()
    pairs = parse_pairs(args.pairs, class_names)
    mean, std = get_norm_stats(get_split()["train_indices"])
    _, cifar_indices = split_labels(dataset)

    run_name = args.runs[0]
    model, _ = load_run(run_name, device)
    _, acc, preds, true_labels = evaluate(model, loader, make_criterion(), device)
    print(f"[{split_name}] acc {acc:.4f} ({len(true_labels):,}장)")
    cm = confusion_matrix(preds, true_labels, config.NUM_CLASSES)

    stem = output_stem("misclassified", split_name, args.runs, args.layer)
    cm_table = pd.DataFrame(cm.numpy(), index=class_names, columns=class_names)
    cm_table.to_csv(OUTPUT_DIR / f"{stem}_confusion.csv")
    save_confusion_matrix_figure(cm_table, f"{run_name}\n{split_name} acc {acc:.4f}",
                                 OUTPUT_DIR / f"{stem}_confusion.png")
    print(f"  → 저장: {(OUTPUT_DIR / f'{stem}_confusion.png').relative_to(config.PROJECT_DIR)}")
    print(f"  → 저장: {(OUTPUT_DIR / f'{stem}_confusion.csv').relative_to(config.PROJECT_DIR)}")

    preds, labels = preds.numpy(), true_labels.numpy()
    rng = np.random.default_rng(config.SEED)
    print("\n클래스 쌍별 오분류 수")
    positions, pair_names = [], []
    for first, second in pairs:
        name_a, name_b = class_names[first], class_names[second]
        print(f"  {name_a} -> {name_b}: {cm[first, second].item()}건 | "
              f"{name_b} -> {name_a}: {cm[second, first].item()}건")
        chosen = select_confused(preds, labels, first, second, args.max_per_pair, rng)
        positions.extend(chosen)
        pair_names.extend([f"{name_a},{name_b}"] * len(chosen))
    if not positions:
        print("지정한 쌍에 오분류 샘플이 없어 샘플 그림은 만들지 않습니다.")
        return

    images = load_images(dataset, positions)
    pred_targets = torch.as_tensor(preds[positions])
    true_targets = torch.as_tensor(labels[positions])
    with GradCAM(model, args.layer) as cam:
        pred_cams, probs, _ = compute_cams(cam, images, device, pred_targets)
        true_cams, _, _ = compute_cams(cam, images, device, true_targets)
        describe_layer(run_name, cam)

    fig, axes = make_grid(len(positions), 3)
    mappable = None
    for row, position in enumerate(positions):
        image = denormalize(images[row], mean, std)
        pred, true = preds[position], labels[position]
        axes[row, 0].imshow(image, interpolation="nearest")
        axes[row, 0].set_title(f"#{position} true: {class_names[true]}\n"
                               f"pred: {class_names[pred]} ({probs[row, pred]:.2f})",
                               fontsize=TITLE_FONT_SIZE)
        mappable = show_heatmap(axes[row, 1], pred_cams[row].numpy())
        axes[row, 1].set_title(f"CAM for pred: {class_names[pred]}", fontsize=TITLE_FONT_SIZE)
        show_heatmap(axes[row, 2], true_cams[row].numpy())
        axes[row, 2].set_title(f"CAM for true: {class_names[true]}", fontsize=TITLE_FONT_SIZE)
    add_colorbar(fig, mappable, axes)

    save_figure(fig, stem)
    save_indices(stem, split_name, args.runs, positions, cifar_indices, labels, class_names,
                 extra={"pair": pair_names, "pred_class": [class_names[preds[p]] for p in positions],
                        "confidence": [probs[row, preds[p]].item() for row, p in enumerate(positions)]})


# =====================================================================
# 모드 check
# =====================================================================
def check_all_models(images, device):
    """1) 모든 등록 모델에서 target layer를 찾고, CAM의 shape·값 범위·NaN을 확인한다."""
    problems = []
    for name in list_models():
        set_seed()
        model = get_model(name).to(device)
        with GradCAM(model) as cam:
            cams, _, _ = cam(images)
            source_size = cam.activation.shape[-1]
            index = cam.layer_index
        valid = (cams.shape == (len(images), config.IMAGE_SIZE, config.IMAGE_SIZE)
                 and not torch.isnan(cams).any().item()
                 and cams.min().item() >= 0 and cams.max().item() <= 1
                 and source_size == EXPECTED_CAM_SOURCE_SIZE
                 and isinstance(model.features[index], nn.ReLU))
        print(f"    {name:22s} features[{index:2d}] CAM 원본 {source_size}×{source_size} "
              f"출력 {tuple(cams.shape)} 범위 {cams.min():.3f}~{cams.max():.3f} "
              f"{'정상' if valid else '이상'}")
        if not valid:
            problems.append(name)
    return not problems, f"{len(list_models())}개 모델 확인, 이상 {problems or '없음'}"


def check_inplace(trained_model, device):
    """2) inplace ReLU를 끄기 전후 logits가 같은지 확인한다. 다르면 disable_inplace_relu의 assert가 실패한다."""
    total = 0
    for name in list_models():
        set_seed()
        model = get_model(name).to(device)
        total += disable_inplace_relu(model)
        assert not any(isinstance(m, nn.ReLU) and m.inplace for m in model.modules())
    # 학습 모델은 앞에서 이미 해제됐을 수 있으므로 logits 비교만 다시 수행한다
    disable_inplace_relu(trained_model)
    return True, f"랜덤 모델 {len(list_models())}개 + 학습 모델 logits 동일, ReLU {total}개 해제"


def check_all_zero(model, loader, device):
    """3) 학습 모델에서 CAM이 전부 0인 이미지 비율을 전체 데이터로 계산한다."""
    n_zero, n_total = 0, 0
    with GradCAM(model) as cam:
        for images, _ in loader:
            cams, _, _ = cam(images.to(device))
            n_zero += (cams.amax(dim=(1, 2)) == 0).sum().item()
            n_total += len(images)
    ratio = n_zero / n_total
    return ratio <= MAX_ALL_ZERO_RATIO, f"{n_zero}/{n_total}장 = {ratio:.2%} (기준 {MAX_ALL_ZERO_RATIO:.0%} 이하)"


def check_target_change(model, images, device):
    """4) 같은 이미지에서 target class를 1순위에서 2순위 클래스로 바꾸면 CAM이 달라지는지 확인한다."""
    with GradCAM(model) as cam:
        first_cams, logits, _ = cam(images)
        second_class = logits.topk(2, dim=1).indices[:, 1]
        second_cams, _, used = cam(images, second_class)
    assert torch.equal(used, second_class), "지정한 target class가 적용되지 않았습니다."
    diff = (first_cams - second_cams).abs().mean().item()
    return diff >= MIN_TARGET_CHANGE_DIFF, f"평균 절대 차이 {diff:.4f} (기준 {MIN_TARGET_CHANGE_DIFF} 이상)"


def check_trained_vs_random(model, model_name, images, device):
    """5) 학습 모델과 랜덤 초기화 모델의 CAM을 비교한다. 거의 같으면 가중치를 반영하지 못하는 구현 오류다."""
    set_seed()
    random_model = get_model(model_name).to(device)
    with GradCAM(model) as cam:
        trained_cams, _, _ = cam(images)
    with GradCAM(random_model) as cam:
        random_cams, _, _ = cam(images)
    diff = (trained_cams - random_cams).abs().mean().item()
    return diff >= MIN_TRAINED_RANDOM_DIFF, f"평균 절대 차이 {diff:.4f} (기준 {MIN_TRAINED_RANDOM_DIFF} 이상)"


def check_val_accuracy(model, info, loader, device):
    """6) 다시 계산한 val 정확도가 체크포인트 기록과 같은지 확인한다. 가중치 로딩과 정규화 점검용이다."""
    _, acc, _, _ = evaluate(model, loader, make_criterion(), device)
    diff = abs(acc - info["val_acc"])
    return diff <= MAX_VAL_ACC_DIFF, (f"다시 계산 {acc:.4f} / 체크포인트 {info['val_acc']:.4f} "
                                      f"(차이 {diff:.4f}, 기준 {MAX_VAL_ACC_DIFF} 이하)")


def check_gap_equivalence(images, device):
    """
    7) gap 계열에서 Grad-CAM과 원래 CAM(FC 가중치 × 특징맵)의 상관계수를 확인한다.

    target layer와 GAP 사이에 MaxPool이 있어도 비례한다. MaxPool은 칸마다 최댓값 위치 하나로만
    gradient를 보내므로, 공간 평균한 채널 가중치가 FC 가중치 w_k에 비례(w_k / 64)하기 때문이다.
    업샘플·정규화 전의 값을 이미지별로 비교한다.
    """
    gap_names, lowest = [], 1.0
    for name in list_models():
        set_seed()
        model = get_model(name).to(device)
        if not any(isinstance(layer, nn.AdaptiveAvgPool2d) for layer in model.classifier):
            continue
        gap_names.append(name)
        with GradCAM(model) as cam:
            raw, _, targets = cam.compute_raw(images)
            activation = cam.activation
        fc_weights = model.classifier[-1].weight[targets].detach()  # (B, C)
        class_cam = (fc_weights[:, :, None, None] * activation).sum(dim=1)
        correlations = [torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
                        for a, b in zip(raw, class_cam)]
        print(f"    {name:22s} 상관계수 최소 {min(correlations):.6f}, 평균 {np.mean(correlations):.6f}")
        lowest = min(lowest, min(correlations))
    passed = bool(gap_names) and lowest >= MIN_GAP_CORRELATION
    return passed, f"{gap_names} 최소 상관계수 {lowest:.6f} (기준 {MIN_GAP_CORRELATION} 이상)"


def run_check(args, device):
    """검증 7개를 실행한다. 하나가 실패해도 나머지를 계속 실행하고, 실패가 있으면 종료 코드 1로 끝낸다."""
    run_name = args.runs[0] if args.runs else CHECK_RUN
    loader, _ = load_split(use_test=False)
    labels, _ = split_labels(loader.dataset)
    images = load_images(loader.dataset, select_per_class(labels, DEFAULT_PER_CLASS)).to(device)
    # 클래스별 이미지보다 많은 첫 배치로 재서, 몇 장의 우연으로 차이가 커지거나 작아지지 않게 한다
    batch_images = next(iter(loader))[0].to(device)
    model, info = load_run(run_name, device)
    print(f"학습 모델: {run_name}\n")

    checks = [
        ("1. 모든 모델 CAM shape·범위·NaN", lambda: check_all_models(images, device)),
        ("2. inplace 해제 전후 logits 동일", lambda: check_inplace(model, device)),
        ("3. CAM 전부 0인 이미지 비율", lambda: check_all_zero(model, loader, device)),
        ("4. target class 변경 시 CAM 변화", lambda: check_target_change(model, batch_images, device)),
        ("5. 학습 vs 랜덤 모델 CAM 차이",
         lambda: check_trained_vs_random(model, info["model_name"], batch_images, device)),
        ("6. val 정확도 재현", lambda: check_val_accuracy(model, info, loader, device)),
        ("7. gap 계열 Grad-CAM = 원래 CAM", lambda: check_gap_equivalence(images, device)),
    ]
    results = []
    for title, check in checks:
        print(f"=== {title} ===")
        try:
            passed, detail = check()
        except (AssertionError, RuntimeError, ValueError) as error:
            passed, detail = False, f"예외 발생: {error}"
        print(f"  [{'PASS' if passed else 'FAIL'}] {detail}\n")
        results.append((title, passed))

    print("=== 요약 ===")
    for title, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {title}")
    if not all(passed for _, passed in results):
        sys.exit(1)


def main():
    """모드에 맞는 함수를 실행한다."""
    args = parse_args()
    set_seed()
    device = config.DEVICE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"장치: {device} | 모드: {args.mode}")
    {"single": run_single, "compare": run_compare,
     "misclassified": run_misclassified, "check": run_check}[args.mode](args, device)


if __name__ == "__main__":
    main()
