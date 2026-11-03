"""
bn_deep_wide_gap: bn_deep_wide의 분류부를 GAP + FC 1개로 교체  →  큰 모델에서 GAP을 다시 확인. (4주차)

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
    GlobalAvgPool                               256×1×1
    Flatten                                     256
    FC(256→10)                                  10 (logits)

어떤 모델과 비교하면 무엇을 알 수 있는가
    - bn_deep_wide_gap vs bn_deep_wide: 같은 특징 추출부에서 분류부만 바꾼 효과
    - 3주차 gap은 Baseline보다 8.4%p 낮았다(20 epoch, 작은 모델).
      특징 추출부가 커져도 GAP의 손해가 큰지 보는 것이 목적이고, 3주차 수치와 직접 비교하지는 않는다.

파라미터 수: 1,148,874개 (bn_deep_wide 2,197,706개 대비 −1,048,832개, 약 48% 감소)
    - ★ 분류부 구조 효과와 모델 크기 효과가 섞여 있다. FC 은닉층을 없애면 파라미터가 거의 절반으로 줄기 때문에,
      정확도가 달라져도 'GAP이라서'인지 '파라미터가 줄어서'인지 이 비교만으로는 나눌 수 없다.
    - 3주차 gap은 파라미터가 약 85% 줄었지만 여기서는 약 48%만 줄고, 남은 파라미터의 99.8%가 특징 추출부에 있다.
      즉 크기 감소 폭이 달라서 3주차와 차이가 나도 원인을 GAP 구조 하나로 돌릴 수 없다.
"""

import config
from . import register_model
from .builder import ConfigurableCNN
from .wide import WIDE_CHANNELS  # 3주차 wide와 같은 정의를 쓰기 위해 값을 다시 적지 않는다


@register_model("bn_deep_wide_gap")
class BNDeepWideGAPCNN(ConfigurableCNN):
    """bn_deep_wide의 분류부를 Global Average Pooling + FC 1개로 교체."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__(num_classes=num_classes, use_bn=True, convs_per_block=2,
                         channels=WIDE_CHANNELS, head="gap")
