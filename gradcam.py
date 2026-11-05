"""
Grad-CAM 직접 구현: 모델이 이미지의 어느 위치를 보고 판단했는지 히트맵으로 계산한다.

원리
    특정 클래스 점수를 특징맵으로 미분하면 "각 채널이 그 클래스 점수에 얼마나 기여하는가"를 알 수 있다.
    이 값을 채널 가중치로 삼아 특징맵을 더하면 클래스와 관련된 위치만 강조된 지도가 된다.

계산 순서
    1) 채널 가중치 = gradient의 공간 평균
    2) 가중합 = Σ (채널 가중치 × 특징맵)
    3) ReLU: 클래스 점수를 높이는 쪽(양수)만 남긴다
    4) bilinear 업샘플링으로 입력 크기(32×32)까지 키운다
    5) 이미지별 0~1 정규화

사용 예:
    with GradCAM(model) as cam:
        cams, logits, targets = cam(images)           # 예측 클래스 기준
        cams, logits, targets = cam(images, labels)   # 지정한 클래스 기준
"""

import torch
import torch.nn.functional as F
from torch import nn

import config

EPS = 1e-8                # 정규화 분모가 0일 때 대신 쓰는 값
INPLACE_PROBE_BATCH = 2   # inplace 해제 전후 logits를 비교할 더미 입력 배치 크기


def find_target_layer(model):
    """
    features 안에서 '마지막 Conv2d 다음에 오는 ReLU'의 인덱스를 찾는다.

    Conv → (BN) → ReLU를 모두 거친 특징맵을 쓰기 위함이다.
    층 번호를 고정하지 않고 찾으므로 deep, bn처럼 구조가 달라도 그대로 동작한다.
    """
    layers = list(model.features)
    conv_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Conv2d)]
    if not conv_indices:
        raise ValueError("features에 Conv2d가 없습니다.")
    for index in range(conv_indices[-1] + 1, len(layers)):
        if isinstance(layers[index], nn.ReLU):
            return index
    raise ValueError("마지막 Conv2d 뒤에 ReLU가 없습니다.")


def disable_inplace_relu(model):
    """
    모델의 모든 ReLU를 메모리에서만 inplace=False로 바꾼다. (파일은 그대로)

    이유: inplace ReLU는 앞 층의 출력 텐서를 그 자리에서 덮어쓴다. 그러면 hook으로 저장한 activation이
    바뀌거나, 역전파할 때 "값이 수정되었다"는 에러가 날 수 있다.
    모델의 계산 결과는 달라지면 안 되므로 전후 logits가 같은지 확인한다.

    Returns:
        inplace를 끈 ReLU 개수
    """
    model.eval()
    device = next(model.parameters()).device
    # 전역 난수 흐름을 건드리지 않도록 독립 generator로 더미 입력을 만든다
    generator = torch.Generator().manual_seed(config.SEED)
    probe = torch.randn(INPLACE_PROBE_BATCH, config.NUM_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE,
                        generator=generator).to(device)

    with torch.no_grad():
        before = model(probe)
    n_changed = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU) and module.inplace:
            module.inplace = False
            n_changed += 1
    with torch.no_grad():
        after = model(probe)
    assert torch.equal(before, after), "inplace ReLU를 끈 뒤 logits가 달라졌습니다."
    return n_changed


class GradCAM:
    """
    forward hook으로 특징맵을, tensor hook으로 gradient를 저장해 Grad-CAM을 계산한다.

    Args:
        model: features(nn.Sequential)를 가진 모델
        target_layer: features 안의 층 인덱스. None이면 find_target_layer로 자동 선택
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        disable_inplace_relu(model)  # hook을 걸기 전에 먼저 해제해야 저장한 값이 안전하다

        self.layer_index = find_target_layer(model) if target_layer is None else target_layer
        if not 0 <= self.layer_index < len(model.features):
            raise ValueError(f"--layer는 0~{len(model.features) - 1} 범위여야 합니다: {self.layer_index}")

        self.activation = None  # 마지막 계산에 쓴 특징맵 (B, C, H, W)
        self.gradient = None    # 마지막 계산에 쓴 gradient (B, C, H, W)
        self._graph_activation = None
        self._hook = model.features[self.layer_index].register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        """forward 때 특징맵을 잡아 두고, 역전파 때 gradient를 받을 tensor hook을 건다."""
        # hook이 걸린 채로 no_grad 안에서 모델을 실행해도(예: engine.evaluate) 에러가 나지 않게 건너뛴다
        if not output.requires_grad:
            return
        self._graph_activation = output
        # module backward hook 대신 tensor hook을 쓴다: '이 층 출력'의 gradient를 직접 받으므로 더 단순하다
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        """역전파가 target layer를 지날 때 gradient를 저장한다."""
        self.gradient = gradient.detach()

    def compute_raw(self, images, target_classes=None):
        """
        정규화 전의 가중합(ReLU 이전)을 계산한다. 원래 CAM과 비교하는 검증에서도 이 값을 쓴다.

        Args:
            images: (B, C, H, W). 모델과 같은 장치에 있어야 한다.
            target_classes: None이면 예측 클래스, 정수면 모든 이미지에 같은 클래스, 텐서면 이미지별 클래스
        Returns:
            raw: (B, H', H') 가중합, logits: (B, num_classes), targets: (B,)
        """
        # eval 모드 필수: train 모드면 Dropout이 매번 다른 뉴런을 끄고, BN이 배치 통계를 써서
        # 이미지 하나의 결과가 같은 배치의 다른 이미지에 영향을 받는다.
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # 호출하는 쪽이 no_grad 안에 있어도 gradient를 계산할 수 있게 한다
        with torch.enable_grad():
            logits = self.model(images)
            if target_classes is None:
                targets = logits.argmax(dim=1)
            else:
                targets = torch.as_tensor(target_classes, device=logits.device, dtype=torch.long)
                if targets.dim() == 0:
                    targets = targets.expand(logits.size(0))
            # softmax 전의 logit을 미분한다: softmax를 거치면 다른 클래스 점수까지 섞여 해석이 흐려진다.
            # 합을 미분해도 eval 모드에서는 이미지끼리 영향이 없어 이미지별 gradient가 그대로 나온다.
            score = logits[torch.arange(logits.size(0), device=logits.device), targets].sum()
            score.backward()

        self.activation = self._graph_activation.detach()
        self._graph_activation = None     # 계산 그래프를 잡고 있지 않도록 해제
        self.model.zero_grad(set_to_none=True)

        weights = self.gradient.mean(dim=(2, 3), keepdim=True)  # 1) 채널 가중치
        raw = (weights * self.activation).sum(dim=1)            # 2) 가중합
        return raw, logits.detach(), targets

    def __call__(self, images, target_classes=None):
        """
        Grad-CAM을 계산한다.

        Returns:
            cams: (B, H, W) 입력 크기, 값 범위 0~1
            logits: (B, num_classes)
            targets: (B,) 실제로 사용한 클래스
        """
        raw, logits, targets = self.compute_raw(images, target_classes)
        cams = F.relu(raw)                                      # 3) 양수 기여만 남김
        cams = F.interpolate(cams.unsqueeze(1), size=images.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze(1)  # 4) 입력 크기로 확대

        # 5) 이미지별 0~1 정규화. 전부 0인 이미지는 분모를 eps로 처리해 NaN 없이 0으로 남긴다.
        low = cams.amin(dim=(1, 2), keepdim=True)
        high = cams.amax(dim=(1, 2), keepdim=True)
        cams = (cams - low) / (high - low).clamp_min(EPS)
        return cams, logits, targets

    def remove(self):
        """hook을 제거한다. 여러 번 호출해도 안전하다."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
