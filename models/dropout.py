"""
dropout: Baseline + FC 층 사이 Dropout(p=0.5)  →  규제(regularization) 효과 확인용.

구조 (입력 3×32×32)
    [Conv3x3(3→32)   → ReLU → MaxPool2] → 32×16×16
    [Conv3x3(32→64)  → ReLU → MaxPool2] → 64×8×8
    [Conv3x3(64→128) → ReLU → MaxPool2] → 128×4×4
    Flatten → FC(2048→256) → ReLU → Dropout(0.5) → FC(256→10)

Baseline 대비 차이
    - 두 FC 층 사이에 Dropout(0.5) 추가 (나머지 동일)
    - Dropout은 파라미터가 없으므로 파라미터 수는 Baseline과 같음 (620,362)
    - 학습 때만 뉴런의 절반을 무작위로 끄고, 평가(model.eval()) 때는 모두 사용
"""

import config
from . import register_model
from .builder import ConfigurableCNN

DROPOUT_P = 0.5  # 이 변형의 정의 자체이므로 모델 파일에 둔다


@register_model("dropout")
class DropoutCNN(ConfigurableCNN):
    """FC 층 사이에 Dropout(p=0.5) 추가."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, dropout=DROPOUT_P)
