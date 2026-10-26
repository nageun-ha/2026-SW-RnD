"""
모델 구조 확인 스크립트 (학습 없이 몇 초 만에 끝남).

실행:
    python model_summary.py

하는 일
    1) 공통 빌더(ConfigurableCNN)를 기본값으로 만들면 BaselineCNN과 완전히 같은지 검사
    2) 등록된 모든 모델의 파라미터 수와 Baseline 대비 변화를 표로 출력 → results/model_summary.csv
"""

import sys

import pandas as pd
import torch

import config
from data import set_seed
from models import MODEL_REGISTRY, count_parameters, get_model, list_models
from models.baseline import BaselineCNN
from models.builder import ConfigurableCNN


def check_builder_matches_baseline():
    """
    ConfigurableCNN() (옵션 전부 기본값)이 BaselineCNN과 같은 모델인지 4단계로 확인한다.

    1) 층 구성: features / classifier의 층 목록(repr)이 글자 하나까지 같은가
    2) 파라미터 수: 둘 다 config.BASELINE_NUM_PARAMS(620,362)인가
    3) 가중치 이름·shape: Baseline 가중치를 빌더에 strict=True로 그대로 불러올 수 있는가
       (strict=True는 이름이 하나라도 다르거나 빠지면 에러 → 층 번호까지 같다는 증거)
    4) 출력: 같은 가중치·같은 입력이면 출력 텐서가 완전히 같은가
    """
    set_seed()
    baseline = BaselineCNN()
    builder = ConfigurableCNN()

    results = {}
    results["층 구성 동일"] = (repr(baseline.features) == repr(builder.features)
                           and repr(baseline.classifier) == repr(builder.classifier))
    results["파라미터 수 동일"] = (count_parameters(baseline) == count_parameters(builder)
                             == config.BASELINE_NUM_PARAMS)
    try:
        builder.load_state_dict(baseline.state_dict(), strict=True)
        results["가중치 이름·shape 동일"] = True
    except RuntimeError as error:
        print(f"  state_dict 불일치: {error}")
        results["가중치 이름·shape 동일"] = False

    if results["가중치 이름·shape 동일"]:
        baseline.eval()
        builder.eval()
        x = torch.randn(config.SUMMARY_BATCH_SIZE, config.NUM_CHANNELS,
                        config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            results["출력 동일"] = torch.equal(baseline(x), builder(x))
    else:
        results["출력 동일"] = False

    for name, ok in results.items():
        print(f"  {'[통과]' if ok else '[실패]'} {name}")
    print(f"  (Baseline {count_parameters(baseline):,} / 빌더 기본값 {count_parameters(builder):,} params)")
    return all(results.values())


def summarize_models():
    """
    등록된 모든 모델의 파라미터 수, Baseline 대비 변화, 출력 shape을 표로 만든다.
    '변경점' 열은 각 모델 클래스 docstring의 첫 줄을 그대로 사용한다.
    """
    x = torch.randn(config.SUMMARY_BATCH_SIZE, config.NUM_CHANNELS,
                    config.IMAGE_SIZE, config.IMAGE_SIZE)
    rows = []
    for name in list_models():
        model = get_model(name).eval()
        with torch.no_grad():
            out_shape = tuple(model(x).shape)
        num_params = count_parameters(model)
        rows.append({
            "model": name,
            "params": num_params,
            "diff_vs_baseline": num_params - config.BASELINE_NUM_PARAMS,
            "ratio": num_params / config.BASELINE_NUM_PARAMS,
            "output_shape": out_shape,
            "description": MODEL_REGISTRY[name].__doc__.strip().splitlines()[0],
        })

    # baseline을 맨 위로, 나머지는 파라미터 수 순
    table = pd.DataFrame(rows)
    table["is_baseline"] = table["model"] == config.MODEL_NAME
    table = (table.sort_values(["is_baseline", "params"], ascending=[False, True])
                  .drop(columns="is_baseline").reset_index(drop=True))

    shown = table.copy()
    shown["params"] = shown["params"].map("{:,}".format)
    shown["diff_vs_baseline"] = shown["diff_vs_baseline"].map("{:+,}".format)
    shown["ratio"] = shown["ratio"].map("{:.2f}x".format)
    print(shown.to_string(index=False))

    shape_ok = all(s == (config.SUMMARY_BATCH_SIZE, config.NUM_CLASSES) for s in table["output_shape"])
    print(f"\n  {'[통과]' if shape_ok else '[실패]'} 모든 모델 출력 shape = "
          f"({config.SUMMARY_BATCH_SIZE}, {config.NUM_CLASSES})")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.RESULTS_DIR / "model_summary.csv", index=False)
    print(f"  → 저장: results/model_summary.csv")
    return shape_ok


def main():
    """동일성 검사와 파라미터 표를 실행한다. 하나라도 실패하면 종료 코드 1."""
    print("=== 1. 빌더 기본값 = BaselineCNN 확인 ===")
    same = check_builder_matches_baseline()

    print("\n=== 2. 등록된 모델 파라미터 수 ===")
    shapes_ok = summarize_models()

    if not (same and shapes_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
