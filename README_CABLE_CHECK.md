# 노트북 케이블 체크 시스템 사용 가이드 📖

이 프로젝트는 YOLOv5를 사용하여 노트북 수리 후 케이블 단자들이 제대로 꼽혀있는지 자동으로 체크하는 시스템입니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치 및 환경 설정
python setup_cable_check.py
```

### 2. 데이터셋 준비
```bash
# 원본 이미지들을 YOLO 형식으로 변환
python prepare_dataset.py --source_dir "path/to/your/images" --output_dir "datasets/laptop_cable_check"
```

### 3. 모델 학습
```bash
# 케이블 체크 모델 학습
python train_cable_check.py --epochs 100 --batch-size 16

# 또는 배치 파일 실행 (Windows)
start_training.bat
```

### 4. 탐지 실행
```bash
# 학습된 모델로 케이블 상태 탐지
python detect_cable_check.py --source "path/to/test/images" --weights "runs/train/cable_check/weights/best.pt"
```

## 📊 지원하는 케이블 타입

| 클래스 ID | 케이블 타입 | 연결 상태 |
|-----------|-------------|-----------|
| 0 | Power Cable | Connected |
| 1 | Power Cable | Disconnected |
| 2 | USB Cable | Connected |
| 3 | USB Cable | Disconnected |
| 4 | HDMI Cable | Connected |
| 5 | HDMI Cable | Disconnected |
| 6 | Ethernet Cable | Connected |
| 7 | Ethernet Cable | Disconnected |
| 8 | Audio Cable | Connected |
| 9 | Audio Cable | Disconnected |

## 📁 디렉토리 구조

```
yolo-check/
├── datasets/
│   └── laptop_cable_check/
│       ├── images/
│       │   ├── train/          # 훈련용 이미지
│       │   ├── val/            # 검증용 이미지
│       │   └── test/           # 테스트용 이미지
│       └── labels/
│           ├── train/          # 훈련용 라벨 (YOLO 형식)
│           ├── val/            # 검증용 라벨
│           └── test/           # 테스트용 라벨
├── yolov5/
│   ├── data/
│   │   └── laptop_cable_check.yaml    # 데이터셋 설정
│   ├── train_cable_check.py            # 학습 스크립트
│   ├── detect_cable_check.py           # 탐지 스크립트
│   ├── setup_cable_check.py            # 환경 설정 스크립트
│   └── prepare_dataset.py              # 데이터셋 준비 스크립트
└── runs/
    ├── train/                          # 학습 결과
    └── detect/                         # 탐지 결과
```

## 🎯 데이터셋 준비 가이드

### 이미지 수집 요구사항:
- **해상도**: 640x640 이상
- **형식**: JPG, PNG
- **수량**: 각 클래스별 최소 50-100장 (총 500-1000장 권장)
- **다양성**: 다양한 각도, 조명, 배경에서 촬영

### 라벨링:
- YOLO 형식 사용 (class x_center y_center width height)
- 좌표는 0-1 사이의 정규화된 값
- 각 이미지마다 동일한 이름의 .txt 파일 필요

### 예시 라벨 파일 (image.txt):
```
0 0.5 0.3 0.2 0.1    # power_cable_connected
2 0.2 0.7 0.15 0.08  # usb_cable_connected
```

## 🔧 학습 파라미터 조정

### 기본 설정:
- **Model**: YOLOv5s (빠른 속도)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Learning Rate**: 자동 조정

### 성능 향상을 위한 팁:
1. **더 많은 데이터**: 클래스별 100장 이상
2. **데이터 증강**: 회전, 밝기 조정, 노이즈 추가
3. **더 큰 모델**: YOLOv5m 또는 YOLOv5l 사용
4. **하이퍼파라미터 튜닝**: `data/hyps/` 폴더의 설정 파일 수정

## 📈 모델 평가

### 학습 중 모니터링:
```bash
# TensorBoard로 학습 과정 모니터링
tensorboard --logdir runs/train
```

### 주요 메트릭:
- **mAP50**: 정확도 (높을수록 좋음)
- **mAP50-95**: 엄격한 정확도
- **Precision**: 정밀도
- **Recall**: 재현율

## 🚨 문제 해결

### 일반적인 문제들:

1. **CUDA 메모리 부족**:
   ```bash
   python train_cable_check.py --batch-size 8  # 배치 크기 줄이기
   ```

2. **학습이 느림**:
   ```bash
   python train_cable_check.py --workers 4  # 워커 수 줄이기
   ```

3. **정확도가 낮음**:
   - 더 많은 데이터 수집
   - 라벨링 품질 확인
   - 더 많은 에포크로 학습

## 📞 지원

문제가 발생하면 다음을 확인해주세요:
1. Python 3.7+ 설치 확인
2. 필요한 패키지 설치 확인: `pip install -r requirements.txt`
3. CUDA 버전 호환성 확인 (GPU 사용 시)

## 🎉 활용 예시

학습된 모델을 사용하여:
- 수리 완료된 노트북의 케이블 연결 상태 자동 체크
- 품질 관리 시스템 구축
- 수리 프로세스 자동화
- 실시간 모니터링 시스템

---

**Happy Coding! 🚀**
