"""
CIFAR-10 탐색적 데이터 분석(EDA).

실행: python eda.py
결과: figures/ 폴더에 그림(.png)과 표(.csv) 저장, 콘솔에 통계 출력

※ 그림 안의 글자는 영어로 쓴다. matplotlib 기본 폰트에 한글이 없어서
   한글을 쓰면 네모(□)로 깨지기 때문이다.
"""

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장 (서버/원격 환경에서도 동작)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from data import (
    build_transforms,
    compute_mean_std,
    get_norm_stats,
    get_split,
    get_train_val_datasets,
    load_cifar10,
    set_seed,
)


def save_figure(fig, filename):
    """그림을 figures/ 폴더에 저장하고 메모리에서 닫는다."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 저장: {path.relative_to(config.PROJECT_DIR)}")


# =====================================================================
# 1. 클래스별 샘플 수 분포
# =====================================================================
def plot_class_distribution(split, train_raw, test_raw):
    """
    train / val / test 각각의 클래스별 샘플 수를 표로 출력하고 막대그래프로 저장한다.

    테스트셋은 '라벨 개수'만 세고 이미지는 보지 않는다.
    (모델 선택에 영향을 주지 않는 정보라 누수가 아님)
    """
    labels = np.asarray(train_raw.targets)
    parts = {
        "train": labels[split["train_indices"]],
        "val": labels[split["val_indices"]],
        "test": np.asarray(test_raw.targets),
    }
    # 행: 클래스, 열: split 인 개수 표
    counts = pd.DataFrame({
        name: pd.Series(part).value_counts().sort_index() for name, part in parts.items()
    })
    counts.index = train_raw.classes
    print(counts.to_string())
    print(f"  합계: {counts.sum().to_dict()}")

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(config.FIG_DIR / "class_distribution.csv")

    # split마다 개수 규모가 크게 달라서(4,500 / 500 / 1,000) 그래프를 따로 그림
    # layout="constrained": 제목·축 라벨이 겹치지 않게 자동 배치
    fig, axes = plt.subplots(1, len(counts.columns), figsize=(15, 4.5), layout="constrained")
    for ax, name in zip(axes, counts.columns):
        ax.bar(counts.index, counts[name], color=config.BAR_COLOR, width=0.6)
        ax.set_title(f"{name} (total {counts[name].sum():,})")
        # 라벨 끝을 눈금에 맞춰 오른쪽 정렬해야 긴 이름끼리 겹치지 않음
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)  # 격자선을 막대 뒤로
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("number of images")
    fig.suptitle("CIFAR-10 class distribution per split")
    save_figure(fig, "class_distribution.png")


# =====================================================================
# 2. 클래스별 샘플 이미지 격자
# =====================================================================
def plot_sample_grid(split, train_raw):
    """train 분할에서 클래스마다 몇 장씩 무작위로 뽑아 격자(행=클래스)로 저장한다."""
    rng = np.random.default_rng(config.SEED)
    labels = np.asarray(train_raw.targets)
    train_idx = np.asarray(split["train_indices"])
    n_cols = config.EDA_SAMPLES_PER_CLASS

    fig, axes = plt.subplots(config.NUM_CLASSES, n_cols, figsize=(n_cols * 1.2, config.NUM_CLASSES * 1.2),
                             layout="constrained")
    for c, class_name in enumerate(train_raw.classes):
        class_idx = train_idx[labels[train_idx] == c]           # train 중 클래스 c만
        chosen = rng.choice(class_idx, size=n_cols, replace=False)
        for j, idx in enumerate(chosen):
            ax = axes[c, j]
            ax.imshow(train_raw.data[idx])                       # 원본 uint8 이미지
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(class_name, rotation=0, ha="right", va="center")
    fig.suptitle("CIFAR-10 samples per class (train split)")
    save_figure(fig, "sample_grid.png")


# =====================================================================
# 3. 정규화 전후 채널별 평균 / 표준편차
# =====================================================================
def channel_stats_from_dataset(dataset):
    """
    정규화된 텐서를 내놓는 Dataset 전체의 채널별 평균/표준편차를 계산한다.

    배치 단위로 합계와 제곱합을 누적해서 메모리를 아낀다.
    """
    loader = DataLoader(dataset, batch_size=config.STATS_BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    channel_sum = torch.zeros(config.NUM_CHANNELS, dtype=torch.float64)
    channel_sq_sum = torch.zeros(config.NUM_CHANNELS, dtype=torch.float64)
    n_pixels = 0
    for images, _ in loader:
        images = images.double()                      # (B, C, H, W)
        channel_sum += images.sum(dim=(0, 2, 3))      # 채널(dim=1)만 남김
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def print_normalization_stats(split, train_raw):
    """
    정규화 전(0~1 스케일)과 후의 채널별 평균/표준편차를 train, val 각각 출력한다.

    - train 정규화 후: train 통계로 정규화했으므로 평균≈0, 표준편차≈1 이어야 정상
    - val 정규화 후: train 통계를 썼으므로 0/1에 '가깝지만 정확히 같지는 않음'
      (이게 정상이며, val 통계를 쓰지 않았다는 증거이기도 함)
    """
    # 정규화 '후' 통계는 증강 없는 transform으로 계산해야 순수한 정규화 효과만 볼 수 있음
    train_set_eval, val_set = get_train_val_datasets(augment=False)

    rows = []
    for split_name, indices, normalized_set in [
        ("train", split["train_indices"], train_set_eval),
        ("val", split["val_indices"], val_set),
    ]:
        before_mean, before_std = compute_mean_std(train_raw.data[indices])
        after_mean, after_std = channel_stats_from_dataset(normalized_set)
        for stage, stat, values in [
            ("before", "mean", before_mean), ("before", "std", before_std),
            ("after", "mean", after_mean), ("after", "std", after_std),
        ]:
            rows.append({"split": split_name, "stage": stage, "stat": stat,
                         **dict(zip(config.CHANNEL_NAMES, values))})

    table = pd.DataFrame(rows).set_index(["split", "stage", "stat"])
    print(table.round(4).to_string())
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.FIG_DIR / "normalization_stats.csv")
    print("  ※ train 'after'는 평균 0 / 표준편차 1, val 'after'는 그 근처 값이면 정상")


# =====================================================================
# 4. 증강 전후 이미지 비교
# =====================================================================
def denormalize(tensor, mean, std):
    """
    정규화된 (C, H, W) 텐서를 화면 표시용 (H, W, C) 0~1 배열로 되돌린다.
    x_원본 = x_정규화 × std + mean
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()  # CHW → HWC (matplotlib 형식)


def plot_augmentation_comparison(split, train_raw):
    """
    원본 이미지 옆에 같은 이미지를 여러 번 증강한 결과를 나란히 저장한다.

    config의 USE_AUGMENTATION 값과 상관없이, 증강이 어떻게 보이는지 확인하는
    용도이므로 항상 증강을 켠 실제 train transform을 사용한다.
    """
    mean, std = get_norm_stats(split["train_indices"])
    train_transform, _ = build_transforms(mean, std, augment=True)

    rng = np.random.default_rng(config.SEED)
    chosen = rng.choice(split["train_indices"], size=config.EDA_AUG_IMAGES, replace=False)
    n_cols = 1 + config.EDA_AUG_VIEWS  # 원본 1장 + 증강 여러 장

    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(n_cols * 1.4, len(chosen) * 1.5),
                             layout="constrained")
    for row, idx in enumerate(chosen):
        original = Image.fromarray(train_raw.data[idx])
        axes[row, 0].imshow(original)
        axes[row, 0].set_ylabel(train_raw.classes[train_raw.targets[idx]],
                                rotation=0, ha="right", va="center")
        for v in range(config.EDA_AUG_VIEWS):
            augmented = train_transform(original)  # 실제 학습에 들어가는 것과 같은 변환
            axes[row, v + 1].imshow(denormalize(augmented, mean, std))
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title("original")
    for v in range(config.EDA_AUG_VIEWS):
        axes[0, v + 1].set_title(f"aug {v + 1}")
    fig.suptitle(f"Augmentation: pad {config.CROP_PADDING} + random crop, "
                 f"horizontal flip (p={config.HFLIP_PROB})")
    save_figure(fig, "augmentation_compare.png")


# =====================================================================
# 실행
# =====================================================================
def main():
    """EDA 전체를 순서대로 실행한다."""
    set_seed()  # 샘플 선택·증강 결과도 매번 같게
    split = get_split()
    train_raw = load_cifar10(train=True)
    test_raw = load_cifar10(train=False)

    print("=== 1. 클래스별 샘플 수 ===")
    plot_class_distribution(split, train_raw, test_raw)

    print("\n=== 2. 클래스별 샘플 이미지 ===")
    plot_sample_grid(split, train_raw)

    print("\n=== 3. 정규화 전후 채널별 평균/표준편차 ===")
    print_normalization_stats(split, train_raw)

    print("\n=== 4. 증강 전후 비교 ===")
    plot_augmentation_comparison(split, train_raw)

    print("\nEDA 완료.")


if __name__ == "__main__":
    main()
