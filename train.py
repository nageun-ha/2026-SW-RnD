"""
모델 학습 스크립트.

실행 예:
    python train.py                                   # config.py 기본값으로 학습
    python train.py --model baseline --epochs 5       # 일부만 바꿔서 학습
    python train.py --optimizer adam --lr 0.001 --no-augment
    python train.py --model bn_deep --epochs 40 --lr 0.1 --scheduler cosine   # 5주차: 스케줄러 사용

결과 (results/<실험이름>/):
    args.json    : 이번 실험에 사용한 설정 전체 (scheduler 포함)
    history.csv  : epoch별 train/val loss·acc, 소요 시간, 그 epoch에 쓴 lr (lr 열은 5주차에 맨 끝에 추가)
    best.pt      : val 정확도가 가장 높았던 시점의 체크포인트

실험 이름: {model}_{optimizer}_lr{lr}_bs{bs}_ep{epochs}_{aug|noaug}[_{scheduler}][_wd{wd}][_mom{momentum}]_seed{seed}
    대괄호 부분은 config.py 기본값과 다를 때만 붙는다. 그래서 기본값으로 돌린 4주차까지의 폴더 이름은 그대로다.
    (예전 이름에는 wd가 없어서, wd만 바꾼 실험이 기존 폴더와 같은 이름이 되는 문제가 있었다)

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
    parser.add_argument("--scheduler", default=config.SCHEDULER, choices=["none", "cosine", "step"],
                        help="학습률 스케줄러 (기본 none = lr 고정, 4주차까지와 동일)")
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
    """
    설정값으로 실험 이름을 만든다. 폴더 이름만 봐도 어떤 실험인지 알 수 있게 하기 위함.

    scheduler, weight decay, momentum은 config.py 기본값과 다를 때만 이름에 붙인다.
      - 기본값이면 붙이지 않으므로 4주차까지의 폴더 이름이 그대로 유지된다 (기존 결과를 다시 쓸 수 있음).
      - 다르면 반드시 붙이므로 wd만 바꾼 실험이 기존 폴더를 '완료된 실험'으로 착각하거나 덮어쓰지 않는다.
    seed를 맨 끝에 두는 이유: 같은 설정의 시드 3개 폴더가 이름순으로 나란히 정렬된다.
    getattr을 쓰는 이유: run_experiments.py처럼 일부 값만 넘기는 호출도 기본값으로 처리하기 위함.
    """
    aug = "aug" if args.augment else "noaug"
    options = ""
    scheduler = getattr(args, "scheduler", config.SCHEDULER)
    if scheduler != config.SCHEDULER:
        options += f"_{scheduler}"
    weight_decay = getattr(args, "weight_decay", config.WEIGHT_DECAY)
    if weight_decay != config.WEIGHT_DECAY:
        options += f"_wd{weight_decay}"
    momentum = getattr(args, "momentum", config.MOMENTUM)
    if momentum != config.MOMENTUM:  # 이번에는 튜닝하지 않지만, wd와 같은 이름 충돌을 미리 막는다
        options += f"_mom{momentum}"
    return (f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
            f"_ep{args.epochs}_{aug}{options}_seed{args.seed}")


def build_scheduler(name, optimizer, epochs):
    """
    학습률 스케줄러를 만든다. epoch이 끝날 때마다 한 번씩 step()을 부른다.

    - none  : None을 돌려준다. 학습 루프가 스케줄러를 아예 건드리지 않아 기존 동작과 완전히 같다.
    - cosine: lr을 코사인 곡선으로 0 근처까지 서서히 줄인다 (T_max = 전체 epoch).
              마지막 epoch의 lr은 0이 아니라 처음 값의 약 0.15% (40 epoch 기준)다.
    - step  : 전체 epoch의 50%, 75% 지점이 지나면 lr × 0.1 (40 epoch → 21~30 epoch은 0.1배, 31~40 epoch은 0.01배)
    """
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        # epoch가 아주 적으면 두 지점이 같은 값이 될 수 있어 중복을 없앤다 (0 epoch 지점은 쓰지 않음)
        milestones = sorted({max(1, int(epochs * ratio)) for ratio in config.STEP_MILESTONES})
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=config.STEP_GAMMA)
    raise ValueError(f"지원하지 않는 scheduler: {name} (none, cosine, step 중 선택)")


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
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    num_params = count_parameters(model)
    save_args(args, run_dir, num_params)

    print(f"실험: {run_dir.name}")
    print(f"장치: {device} | 모델: {args.model} ({num_params:,} params) | "
          f"train {len(train_loader.dataset):,} / val {len(val_loader.dataset):,}\n")

    history = []
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        # 이번 epoch 동안 쓸 lr을 학습 '전에' 읽어 둔다 (scheduler.step()은 epoch이 끝난 뒤에 lr을 바꾸므로)
        lr = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "lr": lr,  # 5주차 추가. 기존 열 순서를 바꾸지 않도록 맨 끝에 둔다
        })
        if scheduler is not None:
            scheduler.step()  # epoch 단위 스케줄러: 다음 epoch에 쓸 lr로 갱신
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
              f"{epoch_time:5.1f}s"
              # 스케줄러를 쓸 때만 lr을 보여 준다 (기본 실행의 출력 형식은 그대로 유지)
              f"{f' | lr {lr:.5f}' if scheduler is not None else ''}"
              f"{'  *best' if is_best else ''}")

    best = max(history, key=lambda h: h["val_acc"])
    print(f"\n학습 완료. best val acc {best['val_acc']:.4f} (epoch {best['epoch']})")
    print(f"결과 저장 위치: {run_dir}")


if __name__ == "__main__":
    main()
