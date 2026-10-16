"""
Baseline CNN: 이후 모든 모델 비교의 기준점이 되는 가장 단순한 CNN.

구조 (입력 3×32×32 기준)
    [Conv3x3(3→32)   → ReLU → MaxPool2] → 32×16×16
    [Conv3x3(32→64)  → ReLU → MaxPool2] → 64×8×8
    [Conv3x3(64→128) → ReLU → MaxPool2] → 128×4×4
    Flatten → FC(2048→256) → ReLU → FC(256→10)

BatchNorm·Dropout은 일부러 넣지 않았다.
3~4주차에 "BN/Dropout을 추가하면 얼마나 좋아지는가"를 비교할 기준이 되도록 최대한 단순하게 둔다.
"""

from torch import nn

import config
from . import register_model


@register_model("baseline")
class BaselineCNN(nn.Module):
    """
    Conv-ReLU-Pool 블록 3개 + 완전연결층 2개로 이루어진 기본 CNN.

    구조 관련 숫자(채널 수, 은닉층 크기)는 이 모델만의 설계값이므로
    config.py가 아니라 생성자 기본값으로 둔다. → 모델 파일 하나만 보면 구조를 다 알 수 있음.

    Args:
        num_classes: 출력 클래스 수
        channels: 각 Conv 블록의 출력 채널 수
        hidden_dim: 첫 번째 FC 층의 출력 크기
    """

    def __init__(self, num_classes=config.NUM_CLASSES, channels=(32, 64, 128), hidden_dim=256):
        super().__init__()

        # --- 특징 추출부: Conv → ReLU → MaxPool 블록 반복 ---
        layers = []
        in_ch = config.NUM_CHANNELS
        for out_ch in channels:
            layers += [
                # padding=1: 3x3 커널을 써도 가로세로 크기가 그대로 유지됨 (32→32)
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # 2x2 최대 풀링: 크기를 절반으로 줄임 (32→16→8→4)
                nn.MaxPool2d(kernel_size=2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # 풀링을 블록 수만큼 거쳤으므로 최종 크기 = 32 / 2^블록수 = 4
        feature_size = config.IMAGE_SIZE // (2 ** len(channels))
        flat_dim = in_ch * feature_size * feature_size  # 128 * 4 * 4 = 2048

        # --- 분류부: 펼친 뒤 FC 층 2개 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (B, 128, 4, 4) → (B, 2048)
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # 마지막 층에는 Softmax를 넣지 않는다.
            # nn.CrossEntropyLoss가 내부에서 log-softmax를 계산하기 때문 (넣으면 두 번 적용됨)
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """x: (B, 3, 32, 32) → 클래스별 점수(logits): (B, num_classes)"""
        return self.classifier(self.features(x))
