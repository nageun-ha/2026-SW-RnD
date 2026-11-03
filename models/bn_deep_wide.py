"""
bn_deep_wide: BN + 블록마다 Conv 2개 + 채널 2배  →  깊이와 너비를 함께 늘린 효과 확인용. (4주차 2×2 조합)

구조와 텐서 크기 (Conv는 모두 bias=False, 이유는 builder.py 주석 참고)
    입력                                        3×32×32
    Conv3x3(3→64)    → BN(64)  → ReLU          64×32×32
    Conv3x3(64→64)   → BN(64)  → ReLU          64×32×32
    MaxPool2                                    64×16×16
    Conv3x3(64→128)  → BN(128) → ReLU          128×16×16
    Conv3x3(128→128) → BN(128) → ReLU          128×16×16
    MaxPool2                                    128×8×8
    Conv3x3(128→256) → BN(256) → ReLU          256×8×8
    Conv3x3(256→256) → BN(256) → ReLU          256×8×8
    MaxPool2                                    256×4×4
    Flatten                                     4096
    FC(4096→256) → ReLU                         256
    FC(256→10)                                  10 (logits)

어떤 모델과 비교하면 무엇을 알 수 있는가
    - bn_deep_wide vs bn_wide: 이미 넓은 모델에 깊이를 더한 효과
    - bn_deep_wide vs bn_deep: 이미 깊은 모델에 너비를 더한 효과
    - 위 두 차이를 bn_deep − bn, bn_wide − bn과 비교하면 깊이와 너비의 효과가 단순히 더해지는지 알 수 있다.
    - 이 모델은 bn_deep_wide_dropout, bn_deep_wide_gap의 비교 기준이기도 하다.

파라미터 수: 2,197,706개 (bn 620,586개 대비 +1,577,120개, 3.54배)
    - 파라미터 증가도 더해지지 않는다. 깊이만 +193,984개, 너비만 +802,080개를 더하면 +996,064개인데,
      실제로는 +1,577,120개다. 채널이 2배인 상태에서 추가한 Conv는 파라미터가 약 4배가 되기 때문이다.
    - 그래서 2×2에서 상호작용이 보여도 '구조의 시너지'와 '모델이 그만큼 더 커진 효과'가 섞여 있다.
"""

import config
from . import register_model
from .builder import ConfigurableCNN
from .wide import WIDE_CHANNELS  # 3주차 wide와 같은 정의를 쓰기 위해 값을 다시 적지 않는다


@register_model("bn_deep_wide")
class BNDeepWideCNN(ConfigurableCNN):
    """BN + 블록마다 Conv 2개 + 채널 2배 (2×2 조합: Deep O, Wide O)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True, convs_per_block=2,
                         channels=WIDE_CHANNELS)
