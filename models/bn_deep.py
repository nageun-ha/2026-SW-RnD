"""
bn_deep: BN + 블록마다 Conv3x3 2개  →  BN을 고정했을 때 깊이 효과 확인용. (4주차 2×2 조합)

구조와 텐서 크기 (Conv는 모두 bias=False, 이유는 builder.py 주석 참고)
    입력                                        3×32×32
    Conv3x3(3→32)    → BN(32)  → ReLU          32×32×32
    Conv3x3(32→32)   → BN(32)  → ReLU          32×32×32
    MaxPool2                                    32×16×16
    Conv3x3(32→64)   → BN(64)  → ReLU          64×16×16
    Conv3x3(64→64)   → BN(64)  → ReLU          64×16×16
    MaxPool2                                    64×8×8
    Conv3x3(64→128)  → BN(128) → ReLU          128×8×8
    Conv3x3(128→128) → BN(128) → ReLU          128×8×8
    MaxPool2                                    128×4×4
    Flatten                                     2048
    FC(2048→256) → ReLU                         256
    FC(256→10)                                  10 (logits)

어떤 모델과 비교하면 무엇을 알 수 있는가
    - bn_deep vs bn: BN이 있을 때 깊이(Conv 3개 → 6개)의 효과
    - (bn_deep_wide − bn_wide) vs (bn_deep − bn): 넓은 모델과 좁은 모델에서 깊이 효과가 다른지 (상호작용)
    - 3주차 deep은 20 epoch 결과이므로 40 epoch 결과와 직접 비교하지 않는다.
      epoch 수가 다르면 차이가 구조 때문인지 학습 길이 때문인지 구분할 수 없다.

파라미터 수: 814,570개 (bn 620,586개 대비 +193,984개, 1.31배)
    - 증가분 전부가 Conv·BN에서 나오고 FC는 bn과 같다. 그래서 2×2 중 '순수한 구조 변경'에 가장 가깝다.
    - 그래도 전체의 약 65%(527,114개)는 여전히 FC에 있다. 깊어졌다고 파라미터 대부분이 Conv로 옮겨간 것은 아니다.
"""

import config
from . import register_model
from .builder import ConfigurableCNN


@register_model("bn_deep")
class BNDeepCNN(ConfigurableCNN):
    """BN + 블록마다 Conv3x3 2개 (2×2 조합: Deep O, Wide X)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True, convs_per_block=2)
