#!/usr/bin/env python3
"""
YOLOv5 Cable Check Dataset Preparation Script
노트북 케이블 단자 체크 데이터셋 준비 스크립트

Usage:
    python prepare_dataset.py --source_dir path/to/raw/images --output_dir datasets/laptop_cable_check
"""

import argparse
import os
import shutil
import random
from pathlib import Path
import json
from PIL import Image


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare Cable Check Dataset')
    parser.add_argument('--source_dir', type=str, required=True, help='source directory with raw images')
    parser.add_argument('--output_dir', type=str, default='datasets/laptop_cable_check', help='output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test set ratio')
    parser.add_argument('--resize', type=int, default=640, help='resize images to this size')
    parser.add_argument('--check_labels', action='store_true', help='check if label files exist')
    
    return parser.parse_args()


class DatasetPreparer:
    """데이터셋 준비 클래스"""
    
    def __init__(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 클래스 이름 매핑 (정상 상태만)
        self.class_names = {
            'power_normal': 0,
            'usb_normal': 1,
            'hdmi_normal': 2,
            'ethernet_normal': 3,
            'audio_normal': 4
        }
        
        # 통계 정보
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'class_distribution': {}
        }
    
    def create_directory_structure(self):
        """디렉토리 구조 생성"""
        print("📁 Creating directory structure...")
        
        directories = [
            'images/train',
            'images/val',
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for directory in directories:
            dir_path = self.output_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
    
    def get_image_files(self):
        """이미지 파일 목록 가져오기"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_dir.glob(f'**/*{ext}'))
            image_files.extend(self.source_dir.glob(f'**/*{ext.upper()}'))
        
        return image_files
    
    def resize_image(self, image_path, output_path, size=640):
        """이미지 크기 조정"""
        try:
            with Image.open(image_path) as img:
                # RGB로 변환 (PNG의 경우 RGBA일 수 있음)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 비율 유지하면서 리사이즈
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # 정사각형으로 패딩
                new_img = Image.new('RGB', (size, size), (114, 114, 114))
                new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
                
                new_img.save(output_path, quality=95)
                return True
        except Exception as e:
            print(f"❌ Error resizing {image_path}: {e}")
            return False
    
    def split_dataset(self, image_files):
        """데이터셋 분할"""
        random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
    
    def copy_files(self, file_splits, resize_size=None):
        """파일 복사 및 처리"""
        print("📋 Copying and processing files...")
        
        for split_name, files in file_splits.items():
            print(f"\n📂 Processing {split_name} set ({len(files)} files)...")
            
            for i, file_path in enumerate(files):
                # 진행 상황 표시
                if (i + 1) % 10 == 0 or i == len(files) - 1:
                    print(f"   Progress: {i + 1}/{len(files)}")
                
                # 출력 경로 설정
                output_image_path = self.output_dir / 'images' / split_name / file_path.name
                
                # 이미지 복사 또는 리사이즈
                if resize_size:
                    success = self.resize_image(file_path, output_image_path, resize_size)
                else:
                    try:
                        shutil.copy2(file_path, output_image_path)
                        success = True
                    except Exception as e:
                        print(f"❌ Error copying {file_path}: {e}")
                        success = False
                
                if success:
                    # 라벨 파일 찾기 및 복사
                    label_file = file_path.with_suffix('.txt')
                    if label_file.exists():
                        output_label_path = self.output_dir / 'labels' / split_name / label_file.name
                        try:
                            shutil.copy2(label_file, output_label_path)
                        except Exception as e:
                            print(f"❌ Error copying label {label_file}: {e}")
                    
                    # 통계 업데이트
                    self.stats[f'{split_name}_images'] += 1
                    self.stats['total_images'] += 1
    
    def generate_stats_report(self):
        """통계 리포트 생성"""
        report = {
            'dataset_info': {
                'source_directory': str(self.source_dir),
                'output_directory': str(self.output_dir),
                'split_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio,
                    'test': self.test_ratio
                }
            },
            'statistics': self.stats,
            'class_names': {v: k for k, v in self.class_names.items()}
        }
        
        # JSON 리포트 저장
        report_path = self.output_dir / 'dataset_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 콘솔에 출력
        print("\n" + "="*50)
        print("📊 DATASET PREPARATION REPORT")
        print("="*50)
        print(f"📂 Source: {self.source_dir}")
        print(f"📂 Output: {self.output_dir}")
        print(f"\n📈 Statistics:")
        print(f"   Total Images: {self.stats['total_images']}")
        print(f"   Train: {self.stats['train_images']}")
        print(f"   Validation: {self.stats['val_images']}")
        print(f"   Test: {self.stats['test_images']}")
        print(f"\n💾 Report saved: {report_path}")
        print("="*50)
    
    def prepare(self, resize_size=None):
        """데이터셋 준비 실행"""
        print("🚀 Starting Dataset Preparation")
        print("=" * 50)
        
        # 1. 디렉토리 구조 생성
        self.create_directory_structure()
        
        # 2. 이미지 파일 찾기
        print("\n🔍 Finding image files...")
        image_files = self.get_image_files()
        print(f"✅ Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("❌ No image files found!")
            return False
        
        # 3. 데이터셋 분할
        print("\n✂️  Splitting dataset...")
        file_splits = self.split_dataset(image_files)
        print(f"   Train: {len(file_splits['train'])}")
        print(f"   Val: {len(file_splits['val'])}")
        print(f"   Test: {len(file_splits['test'])}")
        
        # 4. 파일 복사 및 처리
        self.copy_files(file_splits, resize_size)
        
        # 5. 통계 리포트 생성
        self.generate_stats_report()
        
        print("\n✅ Dataset preparation completed successfully!")
        return True


def main():
    """Main function"""
    opt = parse_opt()
    
    # 비율 검증
    total_ratio = opt.train_ratio + opt.val_ratio + opt.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Total ratio must equal 1.0 (current: {total_ratio})")
        return
    
    # 데이터셋 준비 실행
    preparer = DatasetPreparer(
        source_dir=opt.source_dir,
        output_dir=opt.output_dir,
        train_ratio=opt.train_ratio,
        val_ratio=opt.val_ratio,
        test_ratio=opt.test_ratio
    )
    
    resize_size = opt.resize if opt.resize > 0 else None
    preparer.prepare(resize_size)


if __name__ == '__main__':
    main()
