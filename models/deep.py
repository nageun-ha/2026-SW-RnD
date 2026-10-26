"""
deep: 블록마다 Conv3x3 2개 (VGG식)  →  깊이 효과 확인용.

구조 (입력 3×32×32)
    [Conv3x3(3→32)   → ReLU → Conv3x3(32→32)   → ReLU → MaxPool2] → 32×16×16
    [Conv3x3(32→64)  → ReLU → Conv3x3(64→64)   → ReLU → MaxPool2] → 64×8×8
    [Conv3x3(64→128) → ReLU → Conv3x3(128→128) → ReLU → MaxPool2] → 128×4×4
    Flatten → FC(2048→256) → ReLU → FC(256→10)

Baseline 대비 차이
    - 블록당 Conv 1개 → 2개 (Conv 층 3개 → 6개). 블록 수·채널·FC는 동일
    - 3x3 Conv 2개를 쌓으면 receptive field가 5x5와 같아지면서 비선형(ReLU)이 하나 더 들어감
    - 파라미터: 추가된 Conv 3개만큼 증가 (+193,760개)
"""

import config
from . import register_model
from .builder import ConfigurableCNN


@register_model("deep")
class DeepCNN(ConfigurableCNN):
    """블록마다 Conv3x3를 2개씩 쌓은 VGG식 구조."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, convs_per_block=2)
