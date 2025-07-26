#!/usr/bin/env python3
"""
YOLOv5 Cable Check Setup Script (v2.0 Refactored)
노트북 케이블 단자 체크 시스템 설정 스크립트

- 클래스 기반으로 구조 개선
- HEIC 이미지 파일 지원 추가 (`pillow-heif`)
- 가독성 향상

Usage:
    python setup_cable_check.py
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request

class CableCheckSetup:
    """케이블 체크 시스템의 모든 설정 과정을 관리하는 클래스"""

    def __init__(self):
        """설정에 필요한 모든 경로와 정보를 초기화"""
        self.ROOT_DIR = Path(__file__).resolve().parent
        self.WEIGHTS_DIR = self.ROOT_DIR / 'weights'
        
        # 설치할 패키지 목록 (HEIC 지원 추가)
        self.REQUIREMENTS = [
            'torch>=1.7.0', 'torchvision>=0.8.1', 'torchaudio>=0.7.2',
            'opencv-python>=4.1.1', 'Pillow>=7.1.2', 'PyYAML>=5.3.1',
            'requests>=2.23.0', 'scipy>=1.4.1', 'matplotlib>=3.2.2',
            'numpy>=1.18.5', 'tensorboard>=2.4.1', 'pandas>=1.1.4',
            'seaborn>=0.11.0', 'ipython', 'psutil', 'thop',
            'clearml', 'comet_ml', 'pillow-heif'
        ]
        
        # 다운로드할 가중치 정보
        self.WEIGHTS_INFO = [
            ('yolov5s.pt', 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'),
            ('yolov5m.pt', 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt'),
            ('yolov5l.pt', 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt'),
        ]
        
        # 생성할 디렉토리 목록
        self.DIRECTORIES = [
            'datasets/laptop_cable_check/images/train',
            'datasets/laptop_cable_check/images/val',
            'datasets/laptop_cable_check/images/test',
            'datasets/laptop_cable_check/labels/train',
            'datasets/laptop_cable_check/labels/val',
            'datasets/laptop_cable_check/labels/test',
            'runs/train', 'runs/detect'
        ]

    def run(self):
        """설정 스크립트의 메인 실행 함수"""
        self._print_header("🚀 YOLOv5 케이블 체크 시스템 설정 시작!")
        
        self._create_directory_structure()
        self._install_requirements()
        self._download_pretrained_weights()
        self._create_helper_files()
        
        self._print_footer()

    def _install_requirements(self):
        """필요한 파이썬 패키지를 설치"""
        self._print_header("1. 📦 필수 패키지 설치", char='-')
        failed_packages = []
        
        for req in self.REQUIREMENTS:
            try:
                print(f"  -> {req} 설치 중...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✅ {req} 설치 완료!")
            except subprocess.CalledProcessError:
                print(f"  ❌ {req} 설치 실패!")
                failed_packages.append(req)
        
        if 'pillow-heif' not in failed_packages:
            print("\n✨ HEIC 이미지 파일 지원이 활성화되었습니다! 이제 아이폰 사진도 바로 사용할 수 있습니다.")
            
        if failed_packages:
            print(f"\n⚠️ 다음 패키지 설치에 실패했습니다: {', '.join(failed_packages)}")
            print("   수동으로 설치해주세요: pip install <package_name>")
        else:
            print("\n✅ 모든 패키지가 성공적으로 설치되었습니다!")

    def _create_directory_structure(self):
        """프로젝트에 필요한 디렉토리 구조를 생성"""
        self._print_header("2. 📁 폴더 구조 생성", char='-')
        for dir_name in self.DIRECTORIES:
            dir_path = self.ROOT_DIR / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ '{dir_path.relative_to(self.ROOT_DIR)}' 폴더 생성 완료")
        print("\n✅ 모든 폴더가 준비되었습니다!")

    def _download_pretrained_weights(self):
        """YOLOv5 사전 훈련된 모델 가중치를 다운로드"""
        self._print_header("3. ⬇️ AI 모델 가중치 다운로드", char='-')
        self.WEIGHTS_DIR.mkdir(exist_ok=True)
        
        for name, url in self.WEIGHTS_INFO:
            path = self.WEIGHTS_DIR / name
            if not path.exists():
                try:
                    print(f"  -> '{name}' 다운로드 중...")
                    urllib.request.urlretrieve(url, path)
                    print(f"  ✅ '{name}' 다운로드 완료!")
                except Exception as e:
                    print(f"  ❌ '{name}' 다운로드 실패: {e}")
            else:
                print(f"  ✅ '{name}' 이미 존재합니다.")
        print("\n✅ AI 모델이 준비되었습니다!")

    def _create_helper_files(self):
        """데이터셋 가이드와 학습 스크립트 등 보조 파일을 생성"""
        self._print_header("4. 📝 보조 파일 생성", char='-')
        
        # 데이터셋 가이드 README 생성
        readme_content = """
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
"""
        readme_path = self.ROOT_DIR / 'datasets/laptop_cable_check/README.md'
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"  ✅ 데이터셋 가이드 생성 완료: '{readme_path.relative_to(self.ROOT_DIR)}'")

        # 학습 실행 스크립트 생성
        train_command = "python train_cable_check.py --epochs 100 --batch-size 8 --data data/laptop_cable_check.yaml --weights yolov5s.pt --name cable_check"
        if os.name == 'nt':
            script_path = self.ROOT_DIR / 'start_training.bat'
            script_content = f"@echo off\n{train_command}\npause"
        else:
            script_path = self.ROOT_DIR / 'start_training.sh'
            script_content = f"#!/bin/bash\n{train_command}"
            os.chmod(script_path, 0o755)
            
        script_path.write_text(script_content, encoding='utf-8')
        print(f"  ✅ 학습 시작 스크립트 생성 완료: '{script_path.name}'")
        print("\n✅ 모든 보조 파일이 준비되었습니다!")

    def _print_header(self, title, char='='):
        """섹션 제목을 보기 좋게 출력"""
        print(f"\n{char*60}\n{title}\n{char*60}")

    def _print_footer(self):
        """설정 완료 후 다음 단계를 안내"""
        self._print_header("🎉 모든 설정이 완료되었습니다! 🎉", char='*')
        print("다음 단계를 진행하세요:\n")
        print("1. 📸 `datasets/laptop_cable_check/images/train` 폴더에 이미지 넣기")
        print("2. 🏷️ `python simple_labeler.py` 실행해서 라벨링하기")
        print("3. 🏃 `python train_cable_check.py` 실행해서 AI 학습시키기")
        print("4. � `python realtime_cable_check.py` 실행해서 실시간 테스트하기")
        print(f"\n{'='*60}")

if __name__ == '__main__':
    try:
        setup = CableCheckSetup()
        setup.run()
    except Exception as e:
        print(f"\n💥 치명적인 오류 발생: {e}")
        print("설정 과정에 문제가 발생했습니다. 오류 메시지를 확인해주세요.")
