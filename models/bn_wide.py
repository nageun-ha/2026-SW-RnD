"""
bn_wide: BN + 채널 2배 (64, 128, 256)  →  BN을 고정했을 때 너비 효과 확인용. (4주차 2×2 조합)

구조와 텐서 크기 (Conv는 모두 bias=False, 이유는 builder.py 주석 참고)
    입력                                        3×32×32
    Conv3x3(3→64)    → BN(64)  → ReLU          64×32×32
    MaxPool2                                    64×16×16
    Conv3x3(64→128)  → BN(128) → ReLU          128×16×16
    MaxPool2                                    128×8×8
    Conv3x3(128→256) → BN(256) → ReLU          256×8×8
    MaxPool2                                    256×4×4
    Flatten                                     4096
    FC(4096→256) → ReLU                         256
    FC(256→10)                                  10 (logits)

어떤 모델과 비교하면 무엇을 알 수 있는가
    - bn_wide vs bn: BN이 있을 때 너비(채널 2배)의 효과
    - bn_wide vs bn_deep: 파라미터를 Conv를 늘리는 데 쓸지, 채널을 늘리는 데 쓸지 (크기가 달라 참고용)
    - (bn_deep_wide − bn_deep) vs (bn_wide − bn): 깊은 모델과 얕은 모델에서 너비 효과가 다른지 (상호작용)

파라미터 수: 1,422,666개 (bn 620,586개 대비 +802,080개, 2.29배)
    - 증가분의 약 65%(524,288개)는 Conv가 아니라 첫 FC에서 나온다. 마지막 채널이 2배가 되면서
      FC 입력이 2048 → 4096으로 커지기 때문이다. Conv·BN 증가분은 277,792개뿐이다.
    - 그래서 이 모델의 향상을 전부 'Conv 특징이 풍부해진 효과'로 해석하면 안 된다. FC가 커진 효과가 섞여 있다.
"""

import config
from . import register_model
from .builder import ConfigurableCNN
from .wide import WIDE_CHANNELS  # 3주차 wide와 같은 정의를 쓰기 위해 값을 다시 적지 않는다


@register_model("bn_wide")
class BNWideCNN(ConfigurableCNN):
    """BN + 채널 2배 (2×2 조합: Deep X, Wide O)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True, channels=WIDE_CHANNELS)
