"""
CIFAR-10 데이터 파이프라인 (로딩 · 분할 · 전처리 · DataLoader).

다른 주차의 학습 코드에서는 아래처럼 불러다 쓰기만 하면 된다.

    from data import set_seed, get_dataloaders
    set_seed()
    train_loader, val_loader = get_dataloaders()

이 파일을 직접 실행하면(python data.py) 데이터 검증 체크가 돌아간다.
"""

import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


# =====================================================================
# 1. 재현성 (시드 고정)
# =====================================================================
def set_seed(seed=config.SEED):
    """
    파이썬·넘파이·파이토치의 난수 시드를 한 번에 고정한다.

    학습 스크립트 맨 처음에 한 번 호출하면 셔플 순서, 증강, 모델 초기값이
    매번 같게 나온다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GPU 연산을 결정적으로 고정 (속도는 약간 느려지지만 결과가 매번 같아짐)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    DataLoader의 각 worker 프로세스 시드를 고정한다.

    NUM_WORKERS > 0 으로 바꿔도 재현성이 유지되도록 하기 위한 장치이다.
    (PyTorch 공식 문서의 권장 방식)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================================
# 2. 원본 데이터 로딩
# =====================================================================
def load_cifar10(train, transform=None):
    """
    CIFAR-10 데이터셋 객체를 만든다. 파일이 없으면 자동으로 다운로드한다.

    Args:
        train: True면 학습용 50,000장, False면 테스트용 10,000장
        transform: 이미지에 적용할 전처리 (None이면 PIL 이미지 그대로)
    """
    return datasets.CIFAR10(
        root=config.DATA_DIR, train=train, transform=transform, download=True
    )


def get_class_names():
    """CIFAR-10 클래스 이름 리스트를 반환한다. (예: ['airplane', 'automobile', ...])"""
    return load_cifar10(train=True).classes


# =====================================================================
# 3. train / validation 분할
# =====================================================================
def stratified_split(labels, val_size, seed):
    """
    클래스 비율을 유지하면서 인덱스를 train / val로 나눈다 (stratified split).

    각 클래스마다 (전체 대비 그 클래스 비율 × val_size) 개를 무작위로 뽑아
    validation으로 보낸다. CIFAR-10은 클래스당 5,000장으로 균형이므로
    클래스당 정확히 500장씩 validation이 된다.

    Returns:
        (train_indices, val_indices): 정렬된 파이썬 int 리스트 두 개
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)  # 시드 고정된 독립 난수 생성기

    train_idx, val_idx = [], []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]          # 클래스 c에 속한 인덱스들
        n_val = round(val_size * len(class_idx) / len(labels))
        rng.shuffle(class_idx)                         # 클래스 내부에서 섞기
        val_idx.extend(class_idx[:n_val])
        train_idx.extend(class_idx[n_val:])

    # 반올림 때문에 개수가 어긋나는 경우를 방지
    assert len(val_idx) == val_size, f"val 개수 불일치: {len(val_idx)} != {val_size}"

    # JSON 저장을 위해 파이썬 int로 변환하고, 보기 좋게 정렬
    return sorted(int(i) for i in train_idx), sorted(int(i) for i in val_idx)


def get_split():
    """
    train / val 분할 인덱스를 가져온다.

    - splits/ 폴더에 분할 파일이 있으면 그대로 불러온다 (팀원 모두 같은 분할 사용).
    - 없으면 새로 만들어 저장한다.

    Returns:
        dict: {"seed", "val_size", "train_indices", "val_indices"}
    """
    if config.SPLIT_FILE.exists():
        with open(config.SPLIT_FILE, encoding="utf-8") as f:
            split = json.load(f)
        # 설정이 바뀌었는데 예전 파일을 쓰는 실수 방지
        if split["seed"] != config.SEED or split["val_size"] != config.VAL_SIZE:
            raise ValueError(
                f"{config.SPLIT_FILE.name}의 설정(seed={split['seed']}, "
                f"val_size={split['val_size']})이 config.py와 다릅니다."
            )
        return split

    print(f"[분할] {config.SPLIT_FILE.name} 이(가) 없어 새로 생성합니다.")
    labels = load_cifar10(train=True).targets
    train_idx, val_idx = stratified_split(labels, config.VAL_SIZE, config.SEED)
    split = {
        "seed": config.SEED,
        "val_size": config.VAL_SIZE,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f)
    return split


# =====================================================================
# 4. 정규화 통계 (train 데이터에서만 계산)
# =====================================================================
def compute_mean_std(images_uint8):
    """
    uint8 이미지 배열의 채널별 평균/표준편차를 0~1 스케일로 계산한다.

    한꺼번에 float로 바꾸면 메모리를 1GB 넘게 쓰므로,
    STATS_BATCH_SIZE장씩 잘라서 합계와 제곱합을 누적하는 방식으로 계산한다.
    (분산 = 제곱의 평균 - 평균의 제곱)

    Args:
        images_uint8: shape (N, H, W, C), dtype uint8 넘파이 배열
    Returns:
        (mean, std): 각각 길이 C인 파이썬 float 리스트
    """
    channel_sum = np.zeros(config.NUM_CHANNELS)
    channel_sq_sum = np.zeros(config.NUM_CHANNELS)
    n_pixels = 0

    for start in range(0, len(images_uint8), config.STATS_BATCH_SIZE):
        chunk = images_uint8[start:start + config.STATS_BATCH_SIZE]
        chunk = chunk.astype(np.float64) / config.PIXEL_MAX  # 0~1로 변환
        channel_sum += chunk.sum(axis=(0, 1, 2))             # 채널(C)만 남기고 합산
        channel_sq_sum += (chunk ** 2).sum(axis=(0, 1, 2))
        n_pixels += chunk.shape[0] * chunk.shape[1] * chunk.shape[2]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def get_norm_stats(train_indices):
    """
    정규화에 쓸 채널별 평균/표준편차를 가져온다.

    - 반드시 train 인덱스(45,000장)의 이미지로만 계산한다.
      → val/test 정보가 전처리에 섞이는 '데이터 누수'를 막기 위함.
    - 한 번 계산하면 splits/ 폴더에 저장하고 이후에는 불러와서 쓴다.

    Returns:
        (mean, std): 각각 길이 3인 리스트 (R, G, B 순서)
    """
    if config.STATS_FILE.exists():
        with open(config.STATS_FILE, encoding="utf-8") as f:
            stats = json.load(f)
        if stats["seed"] != config.SEED or stats["num_images"] != len(train_indices):
            raise ValueError(f"{config.STATS_FILE.name}이(가) 현재 분할과 맞지 않습니다.")
        return stats["mean"], stats["std"]

    print(f"[통계] {config.STATS_FILE.name} 이(가) 없어 train 데이터로 계산합니다.")
    raw = load_cifar10(train=True)
    train_images = raw.data[train_indices]  # (45000, 32, 32, 3) uint8, train만 선택
    mean, std = compute_mean_std(train_images)

    stats = {
        "seed": config.SEED,
        "computed_on": "train split only",
        "num_images": len(train_indices),
        "mean": mean,
        "std": std,
    }
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return mean, std


# =====================================================================
# 5. Transform (전처리)
# =====================================================================
def build_transforms(mean, std, augment=config.USE_AUGMENTATION):
    """
    train용 / 평가(val, test)용 transform을 각각 만든다.

    - 평가용: 텐서 변환 + 정규화 (무작위 요소 없음 → 항상 같은 결과)
    - train용: (증강 사용 시) 패딩 후 랜덤 크롭 + 좌우반전 → 텐서 변환 + 정규화

    Args:
        mean, std: train 데이터에서 계산한 채널별 평균/표준편차
        augment: False면 train도 평가용과 똑같은 transform 사용 (baseline 실험용)
    Returns:
        (train_transform, eval_transform)
    """
    # 모든 split에 공통으로 적용되는 부분
    to_normalized_tensor = [
        transforms.ToTensor(),                # PIL(0~255) → Tensor(0~1), HWC → CHW
        transforms.Normalize(mean, std),      # 채널별 (x - mean) / std
    ]
    eval_transform = transforms.Compose(to_normalized_tensor)

    if not augment:
        return eval_transform, eval_transform

    # 증강은 PIL 이미지 단계에서 먼저 적용한 뒤 텐서로 변환
    augmentation = [
        transforms.RandomCrop(config.IMAGE_SIZE, padding=config.CROP_PADDING),
        transforms.RandomHorizontalFlip(p=config.HFLIP_PROB),
    ]
    train_transform = transforms.Compose(augmentation + to_normalized_tensor)
    return train_transform, eval_transform


def get_transforms(augment=None):
    """
    분할 파일과 정규화 통계를 읽어 (train_transform, eval_transform)을 바로 만들어 준다.

    Args:
        augment: None이면 config.USE_AUGMENTATION 값을 따른다.
    """
    if augment is None:
        augment = config.USE_AUGMENTATION
    split = get_split()
    mean, std = get_norm_stats(split["train_indices"])
    return build_transforms(mean, std, augment)


# =====================================================================
# 6. Dataset / DataLoader
# =====================================================================
def get_train_val_datasets(augment=None):
    """
    train / validation Dataset을 만든다.

    핵심: transform이 다른 '데이터셋 객체 두 개'를 만들고,
    같은 분할 인덱스를 각각에 Subset으로 적용한다.
      - train_base (train_transform) + train 인덱스 → train_set
      - eval_base  (eval_transform)  + val 인덱스   → val_set
    random_split을 쓰면 두 Subset이 하나의 데이터셋(=하나의 transform)을
    공유해서 validation에도 증강이 적용되는 문제가 생기므로 이렇게 분리한다.

    Returns:
        (train_set, val_set): torch.utils.data.Subset 두 개
    """
    split = get_split()
    train_transform, eval_transform = get_transforms(augment)

    train_base = load_cifar10(train=True, transform=train_transform)
    eval_base = load_cifar10(train=True, transform=eval_transform)

    train_set = Subset(train_base, split["train_indices"])
    val_set = Subset(eval_base, split["val_indices"])
    return train_set, val_set


def get_dataloaders(augment=None):
    """
    학습에 바로 쓸 수 있는 train / validation DataLoader를 만든다.

    - train: 매 epoch 셔플 (시드 고정된 generator 사용 → 셔플 순서도 재현 가능)
    - val: 셔플하지 않음 (평가 결과가 순서에 영향받지 않도록)

    Args:
        augment: None이면 config.USE_AUGMENTATION, True/False로 직접 지정 가능
    Returns:
        (train_loader, val_loader)
    """
    train_set, val_set = get_train_val_datasets(augment)

    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader


def get_test_loader(confirm_final_eval=False):
    """
    테스트 DataLoader를 만든다. **최종 평가 때만 사용한다.**

    실수로 모델 선택·하이퍼파라미터 튜닝에 테스트셋을 쓰지 않도록,
    confirm_final_eval=True를 명시적으로 넘겨야만 동작한다.
    정규화는 train 통계를 그대로 사용한다.
    """
    if not confirm_final_eval:
        raise RuntimeError(
            "테스트셋은 최종 평가 전까지 사용하지 않습니다. "
            "최종 평가라면 get_test_loader(confirm_final_eval=True)로 호출하세요."
        )
    _, eval_transform = get_transforms(augment=False)
    test_set = load_cifar10(train=False, transform=eval_transform)
    return DataLoader(
        test_set,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


# =====================================================================
# 7. 검증 체크 (python data.py 로 실행)
# =====================================================================
def check_no_overlap(split):
    """train / val 인덱스가 겹치지 않고, 합치면 학습 데이터 전체가 되는지 확인한다."""
    train_idx = set(split["train_indices"])
    val_idx = set(split["val_indices"])

    overlap = train_idx & val_idx  # 교집합
    assert len(overlap) == 0, f"train/val 인덱스가 {len(overlap)}개 겹칩니다!"
    assert len(train_idx) == config.TRAIN_SIZE, f"train 개수 오류: {len(train_idx)}"
    assert len(val_idx) == config.VAL_SIZE, f"val 개수 오류: {len(val_idx)}"
    assert train_idx | val_idx == set(range(config.TRAIN_TOTAL)), "누락된 인덱스가 있습니다!"
    print(f"[통과] 인덱스 중복 없음 (train {len(train_idx):,} / val {len(val_idx):,}, 합계 {config.TRAIN_TOTAL:,})")


def check_split_reproducible(split, labels):
    """같은 시드로 다시 분할했을 때 저장된 분할과 완전히 같은지 확인한다."""
    train_idx, val_idx = stratified_split(labels, config.VAL_SIZE, config.SEED)
    assert train_idx == split["train_indices"] and val_idx == split["val_indices"], \
        "같은 시드인데 분할 결과가 다릅니다! (저장된 파일이 다른 코드/시드로 만들어졌을 수 있음)"
    print(f"[통과] 시드 {config.SEED}로 재분할 결과가 저장된 파일과 동일")


def check_class_ratio(split, labels, class_names):
    """전체 / train / val의 클래스 비율이 같은지 표로 출력하고 확인한다."""
    labels = np.asarray(labels)
    parts = {
        "all": labels,
        "train": labels[split["train_indices"]],
        "val": labels[split["val_indices"]],
    }
    # 각 split의 클래스별 비율을 표로 정리
    ratio_table = pd.DataFrame({
        name: pd.Series(part).value_counts(normalize=True).sort_index()
        for name, part in parts.items()
    })
    ratio_table.index = class_names
    print(ratio_table.round(4).to_string())

    max_diff = (ratio_table[["train", "val"]].sub(ratio_table["all"], axis=0)).abs().max().max()
    assert max_diff <= config.RATIO_TOLERANCE, f"클래스 비율 차이가 큽니다: {max_diff:.4f}"
    print(f"[통과] 클래스 비율 유지 (최대 차이 {max_diff:.4f} ≤ {config.RATIO_TOLERANCE})")


def check_val_no_augmentation(val_loader, train_set, split):
    """
    validation 로더에서 나온 이미지에 증강이 적용되지 않았는지 확인한다.

    1) val 로더의 첫 배치를 여러 번 꺼내도 값이 완전히 같아야 한다.
       (증강이 있다면 랜덤 크롭/반전 때문에 매번 달라짐)
    2) val 로더의 첫 이미지가 '원본 이미지 + 평가용 transform' 결과와 정확히 같아야 한다.
    참고로 train 쪽은 같은 샘플을 여러 번 꺼냈을 때 달라지는지 함께 출력한다.
    """
    # 1) 같은 배치를 반복해서 꺼내 비교 (val은 shuffle=False라 순서도 같음)
    batches = [next(iter(val_loader))[0] for _ in range(config.AUG_CHECK_REPEATS)]
    assert all(torch.equal(batches[0], b) for b in batches[1:]), \
        "val 배치가 꺼낼 때마다 달라집니다 → 증강이 적용된 것으로 보입니다!"

    # 2) 원본 이미지에 평가용 transform만 직접 적용한 결과와 비교
    _, eval_transform = get_transforms()
    raw = load_cifar10(train=True)
    first_val_index = split["val_indices"][0]
    expected = eval_transform(Image.fromarray(raw.data[first_val_index]))
    assert torch.equal(batches[0][0], expected), \
        "val 이미지가 평가용 transform 결과와 다릅니다!"
    print(f"[통과] val 로더에 증강 없음 ({config.AUG_CHECK_REPEATS}회 반복 동일 + 평가용 transform과 일치)")

    # 참고: train 샘플은 증강을 켜면 꺼낼 때마다 달라져야 정상
    train_views = [train_set[0][0] for _ in range(config.AUG_CHECK_REPEATS)]
    n_distinct = len({tuple(v.flatten().tolist()) for v in train_views})
    print(f"[참고] train 샘플 1개를 {config.AUG_CHECK_REPEATS}번 꺼냈을 때 서로 다른 결과 {n_distinct}개 "
          f"(증강 {'ON → 1보다 커야 정상' if config.USE_AUGMENTATION else 'OFF → 1이어야 정상'})")


def print_batch_info(loader, name):
    """배치 하나를 꺼내 shape, dtype, 값 범위, 채널별 평균/표준편차를 출력한다."""
    images, labels = next(iter(loader))
    print(f"[{name}] images {tuple(images.shape)} {images.dtype} | labels {tuple(labels.shape)} {labels.dtype}")
    print(f"        값 범위: min {images.min():.3f}, max {images.max():.3f}")
    # (B, C, H, W)에서 채널(dim=1)만 남기고 통계 계산
    print(f"        채널별 평균: {images.mean(dim=(0, 2, 3)).numpy().round(3)}")
    print(f"        채널별 표준편차: {images.std(dim=(0, 2, 3)).numpy().round(3)}")


def run_checks():
    """모든 데이터 검증 체크를 순서대로 실행한다."""
    set_seed()
    print(f"장치: {config.DEVICE} | 증강 사용: {config.USE_AUGMENTATION}\n")

    split = get_split()
    raw = load_cifar10(train=True)
    mean, std = get_norm_stats(split["train_indices"])
    print(f"정규화 통계 (train 45,000장 기준) mean={np.round(mean, 4)}, std={np.round(std, 4)}\n")

    print("=== 1. 인덱스 중복 / 재현성 ===")
    check_no_overlap(split)
    check_split_reproducible(split, raw.targets)

    print("\n=== 2. 클래스 비율 ===")
    check_class_ratio(split, raw.targets, raw.classes)

    print("\n=== 3. validation 증강 여부 ===")
    train_loader, val_loader = get_dataloaders()
    check_val_no_augmentation(val_loader, train_loader.dataset, split)

    print("\n=== 4. 배치 정보 ===")
    print_batch_info(train_loader, "train")
    print_batch_info(val_loader, "val")

    print("\n모든 체크를 통과했습니다.")


if __name__ == "__main__":
    run_checks()
