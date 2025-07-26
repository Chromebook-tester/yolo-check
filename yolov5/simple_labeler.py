#!/usr/bin/env python3
"""
간단한 케이블 라벨링 도구
사진에서 케이블 위치를 클릭으로 표시
"""

import cv2
import os
from pathlib import Path

class SimpleCableLabeler:
    def __init__(self, image_folder):
        self.image_folder = Path(image_folder)
        self.current_image = None
        self.current_image_path = None
        self.image_files = []
        self.current_index = 0
        self.labels = []
        
        # 클래스 선택
        self.classes = {
            '0': 'sub1-DB', '1': 'sub1-MB', '2': 'sub2-DB', '3': 'sub2-MB',
            '4': 'wifi-antena', '5': 'wifi-connect', '6': 'camera', '7': 'lcd',
            '8': 'battery', '9': 'touchpad', '10': 'keyboard', '11': 'speaker'
        }
        self.current_class = 0
        
        # 이미지 파일 찾기
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend(list(self.image_folder.glob(f'*{ext}')))
        
        if not self.image_files:
            print("❌ 이미지 파일이 없어요!")
            return
            
        print(f"📸 {len(self.image_files)}개 이미지 발견!")
        self.load_image()

    def load_image(self):
        """이미지 로드"""
        if self.current_index >= len(self.image_files):
            print("✅ 모든 이미지 완료!")
            return False
            
        self.current_image_path = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(self.current_image_path))
        self.labels = []
        
        # 기존 라벨 파일 로드 (있다면)
        label_path = self.current_image_path.with_suffix('.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        self.labels.append([int(parts[0]), float(parts[1]), 
                                          float(parts[2]), float(parts[3]), float(parts[4])])
        
        print(f"\n📷 이미지: {self.current_image_path.name} ({self.current_index+1}/{len(self.image_files)})")
        print(f"현재 클래스: {self.current_class} ({self.classes[str(self.current_class)]})")
        print("사용법:")
        print("- 마우스로 케이블 클릭해서 박스 그리기")
        print("- 숫자키 0-9: 클래스 변경")
        print("- 's': 저장, 'n': 다음 이미지, 'q': 종료")
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 클릭한 위치에 작은 박스 추가 (간단하게)
            h, w = self.current_image.shape[:2]
            
            # 정규화된 좌표 (YOLO 형식)
            x_center = x / w
            y_center = y / h
            box_w = 0.1  # 기본 박스 크기
            box_h = 0.1
            
            self.labels.append([self.current_class, x_center, y_center, box_w, box_h])
            print(f"✅ {self.classes[str(self.current_class)]} 추가됨!")
            self.draw_labels()

    def draw_labels(self):
        """라벨들을 화면에 그리기"""
        display_img = self.current_image.copy()
        h, w = display_img.shape[:2]
        
        for label in self.labels:
            class_id, x_center, y_center, box_w, box_h = label
            
            # YOLO 좌표를 픽셀 좌표로 변환
            x1 = int((x_center - box_w/2) * w)
            y1 = int((y_center - box_h/2) * h)
            x2 = int((x_center + box_w/2) * w)
            y2 = int((y_center + box_h/2) * h)
            
            # 모든 정상 상태는 초록색
            color = (0, 255, 0)
            
            # 박스 그리기
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, self.classes[str(class_id)], 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 현재 클래스 표시
        cv2.putText(display_img, f"현재: {self.classes[str(self.current_class)]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('케이블 라벨링', display_img)

    def save_labels(self):
        """라벨 저장"""
        if not self.labels:
            print("저장할 라벨이 없어요!")
            return
            
        label_path = self.current_image_path.with_suffix('.txt')
        with open(label_path, 'w') as f:
            for label in self.labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        
        print(f"💾 저장됨: {label_path}")

    def run(self):
        """라벨링 시작"""
        if not self.image_files:
            return
            
        cv2.namedWindow('케이블 라벨링')
        cv2.setMouseCallback('케이블 라벨링', self.mouse_callback)
        
        while True:
            self.draw_labels()
            key = cv2.waitKey(1) & 0xFF
            
            # 숫자키로 클래스 변경
            if key >= ord('0') and key <= ord('9'):
                self.current_class = key - ord('0')
                print(f"클래스 변경: {self.classes[str(self.current_class)]}")
            
            elif key == ord('s'):  # 저장
                self.save_labels()
            
            elif key == ord('n'):  # 다음 이미지
                self.save_labels()
                self.current_index += 1
                if not self.load_image():
                    break
            
            elif key == ord('q'):  # 종료
                break
        
        cv2.destroyAllWindows()
        print("👋 라벨링 완료!")

if __name__ == "__main__":
    print("📋 케이블 라벨링 도구")
    folder = input("이미지 폴더 경로: ").strip()
    
    if os.path.exists(folder):
        labeler = SimpleCableLabeler(folder)
        labeler.run()
    else:
        print("❌ 폴더가 없어요!")
