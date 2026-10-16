"""
학습 / 평가 루프의 핵심 함수 모음.

train.py, evaluate.py, sanity_check.py가 모두 이 함수들을 공유한다.
→ "검증한 코드"와 "실제로 학습에 쓰는 코드"가 항상 같다는 것이 보장된다.
"""

import torch
from torch import nn


def build_optimizer(name, params, lr, momentum, weight_decay):
    """
    이름으로 optimizer를 만든다.

    Args:
        name: "sgd" 또는 "adam"
        params: model.parameters()
        lr: 학습률
        momentum: SGD 모멘텀 (adam에서는 사용하지 않음)
        weight_decay: L2 정규화 강도
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"지원하지 않는 optimizer: {name} (sgd, adam 중 선택)")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    train 데이터 전체를 한 바퀴(1 epoch) 학습한다.

    Returns:
        (평균 loss, 정확도) — 둘 다 파이썬 float
    """
    # train 모드: Dropout/BatchNorm 같은 층이 '학습용'으로 동작하게 함
    # (Baseline에는 없지만, 나중에 추가될 모델을 위해 항상 호출하는 습관을 들인다)
    model.train()

    total_loss, correct, n_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1) 이전 배치의 기울기 초기화 (안 하면 누적됨)
        outputs = model(images)            # 2) 순전파: (B, 10) logits
        loss = criterion(outputs, labels)  # 3) 배치 평균 loss
        loss.backward()                    # 4) 역전파: 각 파라미터의 기울기 계산
        optimizer.step()                   # 5) 기울기 방향으로 파라미터 업데이트

        # loss는 '배치 평균'이므로 배치 크기를 곱해 합계로 누적
        # → 마지막 배치가 작아도 epoch 평균이 정확해짐
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n_samples += batch_size

    return total_loss / n_samples, correct / n_samples


@torch.no_grad()  # 평가 중에는 기울기를 계산하지 않음 → 메모리 절약, 속도 향상
def evaluate(model, loader, criterion, device):
    """
    데이터 전체에 대해 loss, 정확도를 계산하고 예측값도 함께 반환한다.

    Returns:
        (평균 loss, 정확도, 예측 라벨 텐서, 정답 라벨 텐서)
        예측/정답은 CPU 텐서이며 confusion matrix 계산에 사용한다.
    """
    # eval 모드: Dropout 끄기, BatchNorm은 학습 중 저장한 통계 사용
    model.eval()

    total_loss, n_samples = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        n_samples += labels.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    accuracy = (preds == labels).float().mean().item()
    return total_loss / n_samples, accuracy, preds, labels


def confusion_matrix(preds, labels, num_classes):
    """
    confusion matrix를 계산한다. cm[i, j] = 정답이 i인데 j로 예측한 개수.

    (정답, 예측) 쌍을 하나의 숫자 '정답 × 클래스수 + 예측'으로 바꾼 뒤
    bincount로 한 번에 세는 방식이다. (for문으로 하나씩 세는 것보다 훨씬 빠름)
    """
    index = labels * num_classes + preds
    counts = torch.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def make_criterion():
    """분류 문제의 표준 loss인 CrossEntropyLoss를 만든다. (softmax + NLL을 한 번에 계산)"""
    return nn.CrossEntropyLoss()
