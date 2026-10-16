"""
학습 파이프라인 검증 (본 학습 전에 '코드가 제대로 짜였는지' 확인하는 용도).

실행 예:
    python sanity_check.py                       # 4가지 모두 실행
    python sanity_check.py --check overfit       # 하나만 실행
    python sanity_check.py --model vgg_small     # 3주차에 새 모델도 같은 방법으로 검증

검증 항목
    shape     : 입력/출력 shape, 층별·전체 파라미터 수
    init_loss : 학습 전 loss가 ln(10)≈2.30 근처인지
    overfit   : 64장에 과적합시켜 train acc가 ~100%까지 가는지
    overlap   : 실제 학습에 쓰는 train/val 로더의 인덱스가 겹치지 않는지

※ 클래스별 정확도 / confusion matrix는 학습된 모델이 필요하므로 evaluate.py에서 확인한다.
"""

import argparse
import math
import sys
from itertools import islice

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import config
from data import get_dataloaders, get_train_val_datasets, set_seed
from engine import make_criterion
from models import count_parameters, get_model, list_models


def check_shape(model, val_loader, device):
    """
    실제 배치 하나를 모델에 넣어 입력/출력 shape을 확인하고, 파라미터 수를 출력한다.
    출력이 (배치크기, 10)이 아니면 모델 구조(특히 Flatten 뒤 Linear 입력 크기)가 잘못된 것.
    """
    images, _ = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(images.to(device))
    print(f"  입력 shape:  {tuple(images.shape)}")
    print(f"  출력 shape:  {tuple(outputs.shape)}")

    # 층별 파라미터 수 표 (weight와 bias가 각각 한 줄)
    table = pd.DataFrame(
        [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()],
        columns=["layer", "shape", "num_params"],
    )
    print(table.to_string(index=False))
    print(f"  전체 학습 파라미터 수: {count_parameters(model):,}")

    return tuple(outputs.shape) == (images.shape[0], config.NUM_CLASSES)


def check_init_loss(model, val_loader, device):
    """
    학습 전 초기 loss가 이론값 ln(10)≈2.30 근처인지 확인한다.

    이유: 가중치가 무작위로 작게 초기화되면 모델은 10개 클래스를 거의 같은 확률(1/10)로
    예측한다. 이때 cross-entropy = -ln(1/10) = ln(10) ≈ 2.303 이다.
    이보다 훨씬 크면 → 초기 출력이 한쪽으로 치우침 (초기화·정규화·라벨 문제 의심)
    """
    criterion = make_criterion()
    expected = math.log(config.NUM_CLASSES)

    model.eval()
    losses = []
    with torch.no_grad():
        # val 앞쪽 몇 배치만 사용 (전체를 볼 필요 없음)
        for images, labels in islice(val_loader, config.INIT_LOSS_BATCHES):
            outputs = model(images.to(device))
            losses.append(criterion(outputs, labels.to(device)).item())
    init_loss = sum(losses) / len(losses)

    print(f"  초기 loss: {init_loss:.4f} | 이론값 ln({config.NUM_CLASSES}) = {expected:.4f} "
          f"| 허용 오차 ±{config.INIT_LOSS_TOLERANCE}")
    return abs(init_loss - expected) <= config.INIT_LOSS_TOLERANCE


def check_overfit(model_name, device):
    """
    작은 배치(64장) 하나에 과적합시켜 train acc가 ~100%에 도달하는지 확인한다.

    이유: 모델·loss·역전파·optimizer가 올바르게 연결되어 있다면,
    겨우 64장 정도는 '외워서' 전부 맞힐 수 있어야 한다.
    여기서 실패하면 큰 데이터로 학습해도 소용없음 → 학습 코드부터 고쳐야 한다.

    - 증강은 끈다: 매번 이미지가 달라지면 외우기 어려워져 '코드 문제'와 구분이 안 됨
    - 같은 배치를 계속 반복 학습한다 (DataLoader를 다시 돌리지 않음)
    """
    set_seed()
    model = get_model(model_name).to(device)  # 다른 체크와 섞이지 않게 새 모델로 시작

    train_set, _ = get_train_val_datasets(augment=False)
    small_set = Subset(train_set, range(config.OVERFIT_NUM_SAMPLES))
    images, labels = next(iter(DataLoader(small_set, batch_size=config.OVERFIT_NUM_SAMPLES)))
    images, labels = images.to(device), labels.to(device)
    print(f"  샘플 {len(labels)}장, 포함된 클래스 수 {labels.unique().numel()}개")

    criterion = make_criterion()
    # Adam은 학습률에 덜 민감해서 '코드가 맞는지'만 빠르게 볼 때 편하다
    optimizer = torch.optim.Adam(model.parameters(), lr=config.OVERFIT_LR)

    model.train()
    for step in range(1, config.OVERFIT_STEPS + 1):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        if step == 1 or step % config.OVERFIT_LOG_EVERY == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.4f}")

    print(f"  최종 train acc {acc:.4f} (목표 ≥ {config.OVERFIT_TARGET_ACC})")
    return acc >= config.OVERFIT_TARGET_ACC


def check_overlap(train_loader, val_loader):
    """
    '실제 학습에 쓰는 DataLoader'의 train/val 인덱스가 겹치지 않는지 확인한다.

    1주차 체크는 분할 '파일'을 검사했다면, 여기서는 train.py가 실제로 만드는
    로더 객체(Subset.indices)를 직접 검사한다. → 파일은 맞는데 로더를 잘못 만든 경우까지 잡음
    """
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    overlap = train_idx & val_idx
    print(f"  train {len(train_idx):,} / val {len(val_idx):,} / 겹치는 인덱스 {len(overlap)}개")
    return (len(overlap) == 0
            and len(train_idx) == config.TRAIN_SIZE
            and len(val_idx) == config.VAL_SIZE)


def main():
    """선택한 검증을 실행하고 통과/실패를 요약한다. 하나라도 실패하면 종료 코드 1."""
    parser = argparse.ArgumentParser(description="학습 파이프라인 검증")
    parser.add_argument("--check", default="all",
                        choices=["all", "shape", "init_loss", "overfit", "overlap"])
    parser.add_argument("--model", default=config.MODEL_NAME, choices=list_models())
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE
    print(f"장치: {device} | 모델: {args.model}\n")

    train_loader, val_loader = get_dataloaders()
    model = get_model(args.model).to(device)  # 학습 전 상태의 모델

    # 실행할 검증 목록 (이름, 실행 함수)
    checks = {
        "shape": lambda: check_shape(model, val_loader, device),
        "init_loss": lambda: check_init_loss(model, val_loader, device),
        "overfit": lambda: check_overfit(args.model, device),
        "overlap": lambda: check_overlap(train_loader, val_loader),
    }
    selected = checks if args.check == "all" else {args.check: checks[args.check]}

    results = {}
    for name, run in selected.items():
        print(f"=== {name} ===")
        results[name] = run()
        print(f"  → {'[통과]' if results[name] else '[실패]'}\n")

    print("요약: " + ", ".join(f"{n} {'통과' if ok else '실패'}" for n, ok in results.items()))
    if not all(results.values()):
        sys.exit(1)  # 실패가 있으면 종료 코드 1 (자동화 스크립트에서 감지 가능)


if __name__ == "__main__":
    main()
