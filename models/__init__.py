"""
모델 레지스트리: 모델을 '이름(문자열)'으로 고를 수 있게 해 주는 장치.

새 모델 추가 방법 (다른 파일은 고칠 필요 없음)
    1) models/ 폴더에 새 파일을 만든다. 예: models/vgg_small.py
    2) 클래스 위에 @register_model("vgg_small") 을 붙인다.
    3) python train.py --model vgg_small 로 실행한다.

동작 원리
    - register_model: 클래스를 MODEL_REGISTRY 딕셔너리에 {이름: 클래스}로 등록하는 데코레이터
    - 이 파일 맨 아래에서 models/ 폴더의 모든 .py 파일을 자동으로 import 한다.
      import 되는 순간 데코레이터가 실행되면서 등록이 끝난다.
"""

import importlib
import pkgutil

MODEL_REGISTRY = {}  # {"baseline": BaselineCNN, ...}


def register_model(name):
    """
    클래스를 모델 레지스트리에 이름으로 등록하는 데코레이터.

    사용 예:
        @register_model("baseline")
        class BaselineCNN(nn.Module): ...
    """
    def decorator(cls):
        # 같은 이름을 두 번 등록하면 어떤 모델이 쓰일지 헷갈리므로 바로 에러
        if name in MODEL_REGISTRY:
            raise ValueError(f"모델 이름 '{name}'이(가) 이미 등록되어 있습니다.")
        MODEL_REGISTRY[name] = cls
        return cls  # 클래스 자체는 그대로 돌려줌 (평소처럼 사용 가능)
    return decorator


def get_model(name, **kwargs):
    """
    이름으로 모델 객체를 만들어 반환한다.

    Args:
        name: 등록된 모델 이름 (예: "baseline")
        **kwargs: 모델 생성자에 그대로 전달할 인자 (예: num_classes=10)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"'{name}' 모델이 없습니다. 사용 가능한 모델: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """등록된 모델 이름 목록을 알파벳순으로 반환한다."""
    return sorted(MODEL_REGISTRY)


def count_parameters(model):
    """학습되는(requires_grad=True) 파라미터 개수를 센다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# models/ 폴더 안의 모든 모듈을 자동 import → 각 파일의 @register_model 이 실행됨
# (__path__ 는 이 패키지 폴더 경로, __name__ 은 "models")
for _module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_info.name}")
