"""
gap: Flatten+FC 2개 대신 Global Average Pooling + FC 1개  →  분류부 구조 효과 확인용.

구조 (입력 3×32×32)
    [Conv3x3(3→32)   → ReLU → MaxPool2] → 32×16×16
    [Conv3x3(32→64)  → ReLU → MaxPool2] → 64×8×8
    [Conv3x3(64→128) → ReLU → MaxPool2] → 128×4×4
    GlobalAvgPool → 128×1×1 → Flatten(128) → FC(128→10)

Baseline 대비 차이
    - 특징 추출부(Conv 3블록)는 완전히 동일, 분류부만 교체
    - 4×4 위치 정보를 펼치지 않고 채널별 평균 하나로 요약 → 위치 변화에 덜 민감
    - 주의: Baseline 파라미터의 대부분(약 85%)이 FC(2048→256)에 있으므로,
      이걸 없애면 파라미터가 약 1/6로 줄어든다 (94,538개).
      결과 해석 시 "분류부 구조" 효과와 "모델 크기 감소" 효과가 함께 섞여 있음에 유의.
"""

import config
from . import register_model
from .builder import ConfigurableCNN


@register_model("gap")
class GAPCNN(ConfigurableCNN):
    """분류부를 Global Average Pooling + FC 1개로 바꾼 구조."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, head="gap")
