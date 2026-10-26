"""
bn: Baseline + 각 Conv 뒤 BatchNorm2d  →  정규화 효과 확인용.

구조 (입력 3×32×32)
    [Conv3x3(3→32, bias=False)   → BN(32)  → ReLU → MaxPool2] → 32×16×16
    [Conv3x3(32→64, bias=False)  → BN(64)  → ReLU → MaxPool2] → 64×8×8
    [Conv3x3(64→128, bias=False) → BN(128) → ReLU → MaxPool2] → 128×4×4
    Flatten → FC(2048→256) → ReLU → FC(256→10)

Baseline 대비 차이
    - 각 Conv 뒤에 BatchNorm2d 추가 (나머지 동일)
    - Conv bias 제거: BN이 평균을 빼면서 bias를 상쇄하므로 의미가 없고, BN의 β가 같은 역할을 함
    - 파라미터: Conv bias −224개, BN γ·β +448개 → 총 +224개
"""

import config
from . import register_model
from .builder import ConfigurableCNN


@register_model("bn")
class BNCNN(ConfigurableCNN):
    """각 Conv 뒤에 BatchNorm2d 추가 (Conv-BN-ReLU-Pool)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True)
