"""
프로젝트 전체 설정값을 한 곳에서 관리하는 파일.

- 실험 조건을 바꿀 때는 이 파일만 수정한다.
- 팀원 모두가 같은 값을 쓰도록, 개인적으로 바꾼 값은 커밋 전에 꼭 공유한다.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------
SEED = 42  # 모든 난수(분할, 셔플, 증강, 모델 초기화)의 기준 시드

# ---------------------------------------------------------------------
# 경로 (이 파일이 있는 폴더를 기준으로 계산 → 어디서 실행해도 동일)
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"        # CIFAR-10 원본 다운로드 위치
SPLIT_DIR = PROJECT_DIR / "splits"     # 분할 인덱스·정규화 통계 저장 위치 (팀 공유)
FIG_DIR = PROJECT_DIR / "figures"      # EDA 결과 그림 저장 위치

# 분할 인덱스 파일 / 정규화 통계 파일 (시드가 바뀌면 파일도 따로 생김)
SPLIT_FILE = SPLIT_DIR / f"split_seed{SEED}.json"
STATS_FILE = SPLIT_DIR / f"norm_stats_seed{SEED}.json"

# ---------------------------------------------------------------------
# 데이터셋 정보
# ---------------------------------------------------------------------
NUM_CLASSES = 10
IMAGE_SIZE = 32      # CIFAR-10 이미지 한 변 길이 (32x32)
NUM_CHANNELS = 3     # RGB
PIXEL_MAX = 255.0    # uint8 픽셀 최댓값 (0~255 → 0~1 변환에 사용)

# ---------------------------------------------------------------------
# 데이터 분할
# ---------------------------------------------------------------------
TRAIN_TOTAL = 50_000  # CIFAR-10 공식 학습 데이터 수
VAL_SIZE = 5_000      # 학습 데이터 중 validation으로 떼어낼 수 (클래스당 500장)
TRAIN_SIZE = TRAIN_TOTAL - VAL_SIZE  # 45,000

# ---------------------------------------------------------------------
# 전처리 / 증강
# ---------------------------------------------------------------------
USE_AUGMENTATION = True  # False로 바꾸면 증강 없는 baseline 실험
CROP_PADDING = 4         # 랜덤 크롭 전 상하좌우에 붙일 패딩(px)
HFLIP_PROB = 0.5         # 좌우반전 확률

# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256  # 평가 시에는 역전파가 없어 더 크게 잡아도 됨
# Windows/CPU 환경에서는 0이 가장 안전함 (멀티프로세스 관련 오류·재현성 문제 방지)
NUM_WORKERS = 0


def get_device():
    """
    사용할 장치를 cuda → mps → cpu 순서로 자동 선택한다.

    - cuda: NVIDIA GPU (윈도우 노트북)
    - mps : Apple Silicon GPU (맥미니)
    - cpu : 둘 다 없을 때

    cuda는 '있다'만 확인하지 않고 작은 연산을 실제로 돌려본다.
    설치된 PyTorch가 GPU 세대를 지원하지 않으면(예: RTX 50 시리즈 + CUDA 12.6 빌드)
    is_available()은 True인데 연산에서 에러가 나기 때문이다. 이때는 경고 후 다음 장치로 넘어간다.
    """
    if torch.cuda.is_available():
        try:
            (torch.ones(1, device="cuda") * 2).item()  # 실제 GPU 연산 테스트
            return torch.device("cuda")
        except RuntimeError as error:
            print(f"[경고] CUDA GPU를 찾았지만 연산에 실패해 사용하지 않습니다: {error}".splitlines()[0])
            print("       GPU에 맞는 PyTorch 버전을 설치하세요: https://pytorch.org/get-started/locally/")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
# pin_memory는 CPU→CUDA 전송을 빠르게 하는 옵션이라 cuda일 때만 켠다 (mps/cpu에선 효과 없음)
PIN_MEMORY = DEVICE.type == "cuda"

# ---------------------------------------------------------------------
# 학습 기본 하이퍼파라미터 (train.py 커맨드라인 인자의 기본값)
# ---------------------------------------------------------------------
RESULTS_DIR = PROJECT_DIR / "results"  # 실험 결과 폴더 (실험마다 하위 폴더 생성)
MODEL_NAME = "baseline"
EPOCHS = 20
OPTIMIZER = "sgd"        # "sgd" 또는 "adam"
LR = 0.01
MOMENTUM = 0.9           # SGD에서만 사용
WEIGHT_DECAY = 5e-4      # L2 정규화 강도

# ---------------------------------------------------------------------
# 파이프라인 검증 (sanity_check.py)
# ---------------------------------------------------------------------
INIT_LOSS_BATCHES = 5          # 초기 loss를 평균낼 val 배치 수
INIT_LOSS_TOLERANCE = 0.3      # 초기 loss가 ln(10)≈2.30 에서 이 이상 벗어나면 실패
OVERFIT_NUM_SAMPLES = 64       # 과적합 테스트에 쓸 train 샘플 수
OVERFIT_STEPS = 300            # 같은 배치로 반복 학습할 횟수
OVERFIT_LR = 1e-3              # 과적합 테스트용 Adam 학습률
OVERFIT_TARGET_ACC = 0.99      # 이 정확도 이상이면 통과
OVERFIT_LOG_EVERY = 50         # 몇 step마다 진행 상황을 출력할지

# ---------------------------------------------------------------------
# 모델 구조 확인 (model_summary.py)
# ---------------------------------------------------------------------
BASELINE_NUM_PARAMS = 620_362  # 2주차 BaselineCNN 파라미터 수 (빌더 기본값도 이와 같아야 함)
SUMMARY_BATCH_SIZE = 2         # 출력 shape 확인용 더미 입력 배치 크기

# ---------------------------------------------------------------------
# 실험 일괄 실행 (run_experiments.py) — 예상 시간 계산용
# ---------------------------------------------------------------------
DEFAULT_EPOCH_SEC = 17.0  # 측정값이 하나도 없을 때 쓰는 epoch당 추정 시간 (RTX 5060 baseline 기준)
RUN_STARTUP_SEC = 5.5     # 실험 하나당 시작 비용 (파이썬·데이터셋 로딩, 결과 저장). RTX 5060 실측 약 5.4초

# ---------------------------------------------------------------------
# 검증 체크 / EDA
# ---------------------------------------------------------------------
RATIO_TOLERANCE = 0.001        # 클래스 비율 허용 오차 (0.1%p)
AUG_CHECK_REPEATS = 5          # 같은 샘플을 몇 번 꺼내서 증강 여부를 확인할지
STATS_BATCH_SIZE = 1_000       # 정규화 후 통계 계산 시 배치 크기
EDA_SAMPLES_PER_CLASS = 8      # 클래스별 샘플 격자에서 클래스당 보여줄 이미지 수
EDA_AUG_VIEWS = 7              # 증강 비교 그림에서 원본 1장당 증강 버전 수
EDA_AUG_IMAGES = 4             # 증강 비교 그림에 사용할 원본 이미지 수
FIG_DPI = 150                  # 저장 그림 해상도
BAR_COLOR = "#2a78d6"          # 클래스 분포 막대 색 (단일 계열이라 한 색만 사용)
CHANNEL_NAMES = ["R", "G", "B"]  # 통계 표에 표시할 채널 이름
