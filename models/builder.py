"""
3주차 구조 비교용 공통 빌더: 옵션만 바꿔서 CNN 변형을 만드는 클래스.

각 변형 파일(bn.py, dropout.py, deep.py, wide.py, gap.py)은 이 클래스를 상속하고
옵션 하나만 다르게 넘긴다. → "Baseline과 딱 한 가지만 다른" 비교가 코드 구조로 보장됨.

이 파일은 @register_model을 쓰지 않는다 (학습 대상 모델이 아니라 '부품'이기 때문).

★ 핵심 약속: 옵션을 모두 기본값으로 두면 models/baseline.py의 BaselineCNN과
  층 구성·층 번호·파라미터 수(620,362)가 완전히 같다.
  → model_summary.py가 가중치를 서로 옮겨 넣고 출력까지 같은지 검사한다.
"""

from torch import nn

import config


class ConfigurableCNN(nn.Module):
    """
    옵션으로 구조를 바꿀 수 있는 CNN. 기본값 = BaselineCNN과 동일한 구조.

    기본 구조 (입력 3×32×32)
        [Conv3x3(3→32)   → ReLU → MaxPool2] → 32×16×16
        [Conv3x3(32→64)  → ReLU → MaxPool2] → 64×8×8
        [Conv3x3(64→128) → ReLU → MaxPool2] → 128×4×4
        Flatten → FC(2048→256) → ReLU → FC(256→10)

    Args:
        num_classes: 출력 클래스 수
        channels: 블록별 출력 채널 수 (기본 (32, 64, 128))
        convs_per_block: 블록당 Conv 개수 (기본 1, 2면 VGG식)
        use_bn: True면 각 Conv 뒤에 BatchNorm2d (Conv-BN-ReLU)
        dropout: 0보다 크면 FC 층 사이(ReLU 뒤)에 Dropout(p=dropout)
        head: "fc"   = Flatten → FC → ReLU → FC (기본)
              "gap"  = Global Average Pooling → FC 1개
        hidden_dim: head="fc"일 때 첫 번째 FC의 출력 크기
    """

    def __init__(self, num_classes=config.NUM_CLASSES, channels=(32, 64, 128),
                 convs_per_block=1, use_bn=False, dropout=0.0, head="fc", hidden_dim=256):
        super().__init__()
        if head not in ("fc", "gap"):
            raise ValueError(f"head는 'fc' 또는 'gap'만 가능합니다: {head}")
        if head == "gap" and dropout > 0:
            # 조용히 무시하면 '설정했는데 적용 안 된' 실험이 생기므로 에러로 막는다
            raise ValueError("head='gap'에는 FC 층 사이가 없어 dropout을 적용할 수 없습니다.")

        # --- 특징 추출부 ---
        layers = []
        in_ch = config.NUM_CHANNELS
        for out_ch in channels:
            for _ in range(convs_per_block):
                # BN을 쓸 때는 Conv의 bias를 끈다 (bias=False).
                # 이유: BN이 채널별 '배치 평균'을 빼기 때문에, Conv가 더한 상수 bias는
                #       평균과 함께 그대로 빠져서 아무 효과가 없다.
                #       bias 역할은 BN의 β(shift) 파라미터가 대신한다.
                #       → 표현력은 그대로, 쓸모없는 파라미터만 제거 (ResNet 등 표준 관행)
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                in_ch = out_ch  # 블록 안 두 번째 Conv부터는 입력=출력 채널
            layers.append(nn.MaxPool2d(kernel_size=2))  # 블록 끝에서 크기 절반
        self.features = nn.Sequential(*layers)

        # --- 분류부 ---
        if head == "fc":
            feature_size = config.IMAGE_SIZE // (2 ** len(channels))  # 32 / 2^3 = 4
            head_layers = [
                nn.Flatten(),
                nn.Linear(in_ch * feature_size * feature_size, hidden_dim),
                nn.ReLU(inplace=True),
            ]
            # dropout=0이면 층 자체를 넣지 않는다.
            # Dropout(p=0)을 넣으면 동작은 같아도 층 번호가 밀려서 Baseline과 가중치 이름이 달라짐
            if dropout > 0:
                head_layers.append(nn.Dropout(p=dropout))
            head_layers.append(nn.Linear(hidden_dim, num_classes))
        else:  # "gap"
            head_layers = [
                nn.AdaptiveAvgPool2d(1),   # 채널마다 4×4 평균 → (B, C, 1, 1)
                nn.Flatten(),              # → (B, C)
                nn.Linear(in_ch, num_classes),
            ]
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x):
        """x: (B, 3, 32, 32) → logits: (B, num_classes)"""
        return self.classifier(self.features(x))
