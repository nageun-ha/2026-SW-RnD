"""
학습된 모델 평가: 클래스별 정확도 + confusion matrix.

실행 예:
    python evaluate.py --run baseline_sgd_lr0.01_bs128_ep20_aug_seed42        # validation 평가
    python evaluate.py --run baseline_sgd_lr0.01_bs128_ep20_aug_seed42 --test # 6주차 최종 평가 전용!

결과 (results/<실험이름>/):
    val_per_class_acc.csv, val_confusion_matrix.csv, val_confusion_matrix.png
    (--test 이면 파일 이름 앞이 test_)
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
from data import get_class_names, get_dataloaders, get_test_loader
from engine import confusion_matrix, evaluate, make_criterion
from models import get_model


def parse_args():
    """커맨드라인 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="학습된 모델의 클래스별 정확도 / confusion matrix")
    parser.add_argument("--run", required=True, help="results/ 아래 실험 폴더 이름")
    parser.add_argument("--test", action="store_true",
                        help="테스트셋으로 평가 (6주차 최종 평가 때만 사용!)")
    return parser.parse_args()


def load_model(run_dir, device):
    """best.pt 체크포인트를 읽어 모델을 복원한다."""
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)  # 저장한 장치와 달라도 로드 가능
    model = get_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"체크포인트: {checkpoint['model_name']}, epoch {checkpoint['epoch']}, "
          f"val acc {checkpoint['val_acc']:.4f}")
    return model


def save_confusion_matrix_figure(cm_table, title, path):
    """
    confusion matrix를 히트맵 그림으로 저장한다.
    색은 한 가지 색의 진하기(Blues)만 사용 → 값이 클수록 진하게, 직관적으로 읽힘.
    """
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    image = ax.imshow(cm_table.values, cmap="Blues")
    fig.colorbar(image, ax=ax, label="number of images")

    names = cm_table.columns
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    # 각 칸에 개수 표시. 진한 칸은 흰 글씨, 연한 칸은 검은 글씨로 대비 확보
    threshold = cm_table.values.max() / 2
    for i in range(len(names)):
        for j in range(len(names)):
            value = cm_table.values[i, j]
            ax.text(j, i, value, ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")

    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def main():
    """체크포인트를 불러와 평가하고, 클래스별 정확도와 confusion matrix를 출력·저장한다."""
    args = parse_args()
    device = config.DEVICE
    run_dir = config.RESULTS_DIR / args.run

    if args.test:
        print("⚠ 테스트셋 평가입니다. 모델 선택·튜닝이 모두 끝난 6주차 최종 평가에서만 사용하세요.")
        loader = get_test_loader(confirm_final_eval=True)
        split_name = "test"
    else:
        # val 로더만 사용 (val에는 원래 증강이 적용되지 않음)
        _, loader = get_dataloaders(augment=False)
        split_name = "val"

    model = load_model(run_dir, device)
    loss, acc, preds, labels = evaluate(model, loader, make_criterion(), device)
    print(f"[{split_name}] loss {loss:.4f} | acc {acc:.4f} ({len(labels):,}장)\n")

    class_names = get_class_names()
    cm = confusion_matrix(preds, labels, config.NUM_CLASSES)

    # 클래스별 정확도 = 대각선(맞힌 개수) / 행 합계(그 클래스의 전체 개수)
    per_class = pd.DataFrame({
        "class": class_names,
        "correct": cm.diag().tolist(),
        "total": cm.sum(dim=1).tolist(),
    })
    per_class["accuracy"] = per_class["correct"] / per_class["total"]
    print("클래스별 정확도 (낮은 순)")
    print(per_class.sort_values("accuracy").to_string(index=False, float_format="%.4f"))

    cm_table = pd.DataFrame(cm.numpy(), index=class_names, columns=class_names)
    print("\nconfusion matrix (행=정답, 열=예측)")
    print(cm_table.to_string())

    per_class.to_csv(run_dir / f"{split_name}_per_class_acc.csv", index=False)
    cm_table.to_csv(run_dir / f"{split_name}_confusion_matrix.csv")
    save_confusion_matrix_figure(
        cm_table, f"{run_dir.name}\n{split_name} acc {acc:.4f}",
        run_dir / f"{split_name}_confusion_matrix.png",
    )
    print(f"\n결과 저장 위치: {run_dir}")


if __name__ == "__main__":
    main()
