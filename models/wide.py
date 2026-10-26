"""
wide: 채널 2배 (64, 128, 256)  →  너비 효과 확인용.

구조 (입력 3×32×32)
    [Conv3x3(3→64)    → ReLU → MaxPool2] → 64×16×16
    [Conv3x3(64→128)  → ReLU → MaxPool2] → 128×8×8
    [Conv3x3(128→256) → ReLU → MaxPool2] → 256×4×4
    Flatten → FC(4096→256) → ReLU → FC(256→10)

Baseline 대비 차이
    - 블록별 채널 (32, 64, 128) → (64, 128, 256). 블록 수·FC 은닉 크기(256)는 동일
    - 주의: 마지막 채널이 2배가 되므로 첫 FC의 입력도 2048 → 4096으로 자동으로 커진다.
      (채널을 바꾸면 피할 수 없는 연쇄 변화. 파라미터 증가분의 대부분이 이 FC에서 나옴)
    - 파라미터: 약 2.3배 (1,422,218개)
"""

import config
from . import register_model
from .builder import ConfigurableCNN

WIDE_CHANNELS = (64, 128, 256)  # Baseline (32, 64, 128)의 2배


@register_model("wide")
class WideCNN(ConfigurableCNN):
    """채널 수를 Baseline의 2배로 늘린 구조."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, channels=WIDE_CHANNELS)
