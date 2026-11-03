"""
bn_deep_wide_dropout: bn_deep_wide + FC 층 사이 Dropout(0.5)  →  큰 모델에서 규제 효과를 다시 확인. (4주차)

구조와 텐서 크기 (특징 추출부는 bn_deep_wide와 완전히 같음)
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
    Dropout(0.5)  ← bn_deep_wide와 다른 유일한 부분    256
    FC(256→10)                                  10 (logits)

어떤 모델과 비교하면 무엇을 알 수 있는가
    - bn_deep_wide_dropout vs bn_deep_wide: 큰 모델 + 40 epoch 조건에서 Dropout이 도움이 되는지
    - 3주차 dropout vs baseline(20 epoch, 작은 모델)에서는 효과가 시드 편차 안쪽이었다.
      과적합이 커지는 조건에서 결과가 달라지는지 보는 것이 목적이고, 3주차 수치와 직접 비교하지는 않는다.

파라미터 수: 2,197,706개 (bn_deep_wide와 같음. Dropout에는 파라미터가 없음)
    - 크기가 완전히 같아서 이번 추가 모델 중 '크기 효과'가 섞이지 않는 가장 깨끗한 비교다.
    - 다만 train acc는 Dropout이 켜진 상태에서 재므로 val acc보다 낮게 나올 수 있다.
      train-val 격차가 음수여도 과적합이 없다는 뜻으로 단정하지 말고 val acc로 판단한다.
"""

import config
from . import register_model
from .builder import ConfigurableCNN
from .dropout import DROPOUT_P  # 3주차 dropout과 같은 확률을 쓰기 위해 값을 다시 적지 않는다
from .wide import WIDE_CHANNELS


@register_model("bn_deep_wide_dropout")
class BNDeepWideDropoutCNN(ConfigurableCNN):
    """bn_deep_wide + FC 층 사이 Dropout(0.5)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True, convs_per_block=2,
                         channels=WIDE_CHANNELS, dropout=DROPOUT_P)
