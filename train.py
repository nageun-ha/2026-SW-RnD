"""
모델 학습 스크립트.

실행 예:
    python train.py                                   # config.py 기본값으로 학습
    python train.py --model baseline --epochs 5       # 일부만 바꿔서 학습
    python train.py --optimizer adam --lr 0.001 --no-augment

결과 (results/<실험이름>/):
    args.json    : 이번 실험에 사용한 설정 전체
    history.csv  : epoch별 train/val loss·acc, 소요 시간
    best.pt      : val 정확도가 가장 높았던 시점의 체크포인트

※ 테스트셋은 이 스크립트에서 절대 사용하지 않는다 (6주차 evaluate.py --test 전용).
"""

import argparse
import json
import platform
import time

import pandas as pd
import torch

import config
from data import get_dataloaders, set_seed
from engine import build_optimizer, evaluate, make_criterion, train_one_epoch
from models import count_parameters, get_model, list_models


def parse_args():
    """
    커맨드라인 인자를 읽는다. 기본값은 모두 config.py에서 가져온다.
    → 평소엔 config.py를 그대로 쓰고, 실험마다 바꿀 값만 인자로 덮어쓴다.
    """
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN 학습")
    parser.add_argument("--model", default=config.MODEL_NAME, choices=list_models())
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--optimizer", default=config.OPTIMIZER, choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--momentum", type=float, default=config.MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="가중치 초기화·셔플·증강용 시드 (train/val 분할은 바뀌지 않음)")
    # --augment / --no-augment 두 가지 형태를 자동으로 만들어 줌
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=config.USE_AUGMENTATION)
    parser.add_argument("--run-name", default=None,
                        help="결과 폴더 이름 (생략하면 설정값으로 자동 생성)")
    parser.add_argument("--overwrite", action="store_true",
                        help="같은 이름의 결과 폴더가 있어도 덮어쓰기")
    return parser.parse_args()


def make_run_name(args):
    """설정값으로 실험 이름을 만든다. 폴더 이름만 봐도 어떤 실험인지 알 수 있게 하기 위함."""
    aug = "aug" if args.augment else "noaug"
    return (f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
            f"_ep{args.epochs}_{aug}_seed{args.seed}")


def prepare_run_dir(args):
    """결과 폴더를 만든다. 이미 학습 기록이 있으면 실수로 덮어쓰지 않도록 멈춘다."""
    run_dir = config.RESULTS_DIR / (args.run_name or make_run_name(args))
    if (run_dir / "history.csv").exists() and not args.overwrite:
        raise SystemExit(
            f"이미 결과가 있습니다: {run_dir}\n"
            "다른 --run-name을 쓰거나, 덮어쓰려면 --overwrite를 붙이세요."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_args(args, run_dir, num_params):
    """
    이번 실험의 설정과 실행 환경을 args.json으로 저장한다.
    장치(cuda/mps/cpu)도 기록 → 팀원 간 결과가 미세하게 다를 때 원인을 추적할 수 있음.
    """
    info = {
        **vars(args),
        "device": str(config.DEVICE),
        "num_params": num_params,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    """학습 전체 흐름: 준비 → epoch 반복(학습·검증·기록·체크포인트) → 요약 출력."""
    args = parse_args()
    set_seed(args.seed)  # 모델 생성 '전에' 호출해야 가중치 초기값이 고정됨
    device = config.DEVICE
    run_dir = prepare_run_dir(args)

    # --- 데이터: 테스트셋은 불러오지 않는다 ---
    train_loader, val_loader = get_dataloaders(
        augment=args.augment, batch_size=args.batch_size, seed=args.seed
    )

    # --- 모델 / loss / optimizer ---
    model = get_model(args.model).to(device)
    criterion = make_criterion()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                args.momentum, args.weight_decay)
    num_params = count_parameters(model)
    save_args(args, run_dir, num_params)

    print(f"실험: {run_dir.name}")
    print(f"장치: {device} | 모델: {args.model} ({num_params:,} params) | "
          f"train {len(train_loader.dataset):,} / val {len(val_loader.dataset):,}\n")

    history = []
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        })
        # 매 epoch 저장 → 중간에 멈춰도 그때까지의 기록이 남음
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        # best 체크포인트: 테스트가 아니라 'val 정확도' 기준으로 고른다
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "model_name": args.model,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),  # 나중에 이어서 학습할 때 필요
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": vars(args),
            }, run_dir / "best.pt")

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{epoch_time:5.1f}s{'  *best' if is_best else ''}")

    best = max(history, key=lambda h: h["val_acc"])
    print(f"\n학습 완료. best val acc {best['val_acc']:.4f} (epoch {best['epoch']})")
    print(f"결과 저장 위치: {run_dir}")


if __name__ == "__main__":
    main()
