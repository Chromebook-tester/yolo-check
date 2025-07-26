
# 노트북 케이블 체크 데이터셋 가이드

## 1. 이미지 준비
- `datasets/laptop_cable_check/images/train` 폴더에 훈련용 이미지를 넣어주세요.
- (선택) `images/val` 폴더에 검증용 이미지를 넣어주세요.
- **JPG, PNG, HEIC 등 다양한 이미지 형식을 지원합니다.**

## 2. 라벨링 실행
- 터미널에서 `python simple_labeler.py`를 실행하여 라벨링을 시작하세요.
- 라벨링 결과는 `labels/train` 또는 `labels/val`에 자동으로 저장됩니다.

## 3. 학습 시작
- `start_training.bat` (Windows) 또는 `start_training.sh` (Mac/Linux)를 실행하거나,
- 터미널에서 `python train_cable_check.py`를 실행하세요.
